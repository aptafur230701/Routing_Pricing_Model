"""
attention_decoder.py
====================
Pointer decoder para el agente AM-Actor-Critic.

Recibe el vector de contexto h_t (de ContextNetwork) y los embeddings de nodos
(de AttentionEncoder) y produce una distribución de probabilidad sobre los nodos
válidos, de la que se samplea o se toma el argmax para elegir el destino.

Flujo interno
-------------
1. GlimpseAttention  — cross-attention (h_t → embeddings) → h_t refinado
2. PointerAttention  — compatibilidad dot-product + tanh(C) → logits por nodo
3. apply_mask        — logits de acciones inválidas → -inf
4. Distribución      — softmax(logits) → π(a|s)

Modos de uso
------------
· Entrenamiento (PPO): act() retorna (acción, log_prob, entropía).
  evaluate_action() recalcula log_prob y entropía para un batch de transiciones.
· Evaluación (greedy): greedy_action() retorna el nodo con mayor logit válido.

Compatibilidad
--------------
· La máscara de acción es la misma que produce RoutingEnv._get_action_mask():
  np.int8 array con 1 = válido, 0 = inválido.
· Sin dependencias de agent.py, training.py ni routing_env.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from debug_utils import check_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Bloque 1: Glimpse — cross-attention para refinar el contexto
# ─────────────────────────────────────────────────────────────────────────────

class GlimpseAttention(nn.Module):
    """
    Cross-attention multi-cabeza: el contexto h_t consulta los embeddings
    de nodos y produce un contexto refinado h_t'.

    Query  : h_t            — vector de contexto (batch, d_h)
    Keys   : embeddings     — (batch, N, d_h)
    Values : embeddings     — (batch, N, d_h)
    Salida : h_t'           — (batch, d_h)

    Parámetros
    ----------
    d_h     : int — dimensión del embedding.
    n_heads : int — cabezas de atención (debe dividir d_h).
    """

    def __init__(self, d_h: int, n_heads: int):
        super().__init__()
        assert d_h % n_heads == 0, "d_h debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_k     = d_h // n_heads

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h, bias=False)

    def forward(
        self,
        context:    torch.Tensor,   # (batch, d_h)
        embeddings: torch.Tensor,   # (batch, N, d_h)
        mask:       torch.Tensor,   # (batch, N) bool — True = inválido
    ) -> torch.Tensor:
        """Retorna el contexto refinado h_t' de forma (batch, d_h)."""
        B, N, _ = embeddings.shape
        H, dk   = self.n_heads, self.d_k

        # Query desde el contexto: (batch, 1, d_h) → (batch, H, 1, dk)
        Q = self.W_q(context).view(B, 1, H, dk).transpose(1, 2)
        # Keys/Values desde embeddings: (batch, N, d_h) → (batch, H, N, dk)
        K = self.W_k(embeddings).view(B, N, H, dk).transpose(1, 2)
        V = self.W_v(embeddings).view(B, N, H, dk).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)  # (B,H,1,N)

        # Enmascarar nodos inválidos antes del softmax
        if mask is not None:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2),  # (B,1,1,N)
                float("-inf")
            )

        attn = F.softmax(scores, dim=-1)          # (B, H, 1, N)
        check_tensor("GlimpseAttention attn", attn)
        out  = torch.matmul(attn, V)              # (B, H, 1, dk)
        out  = out.squeeze(2).transpose(1, 2).contiguous().view(B, -1)  # (B, d_h)
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────────────────
# Bloque 2: Pointer — logits finales con clipping tanh
# ─────────────────────────────────────────────────────────────────────────────

class PointerAttention(nn.Module):
    """
    Atención de puntero: compatibilidad entre el contexto refinado h_t' y
    los embeddings de nodos, con clipping tanh (C · tanh(score/sqrt(dk))).

    La constante C (default 10) controla la entropía de la distribución:
    valores altos producen distribuciones más concentradas.

    Entrada : h_t' (batch, d_h), embeddings (batch, N, d_h)
    Salida  : logits sin enmascarar (batch, N)
    """

    def __init__(self, d_h: int, clip_C: float = 10.0):
        super().__init__()
        self.clip_C = clip_C
        self.W_q    = nn.Linear(d_h, d_h, bias=False)
        self.W_k    = nn.Linear(d_h, d_h, bias=False)
        self.d_k    = d_h

    def forward(
        self,
        context:    torch.Tensor,   # (batch, d_h)
        embeddings: torch.Tensor,   # (batch, N, d_h)
    ) -> torch.Tensor:
        """Retorna logits sin enmascarar de forma (batch, N)."""
        q = self.W_q(context).unsqueeze(1)          # (batch, 1, d_h)
        k = self.W_k(embeddings)                     # (batch, N, d_h)
        scores = torch.bmm(q, k.transpose(1, 2))     # (batch, 1, N)
        scores = scores.squeeze(1) / (self.d_k ** 0.5)   # (batch, N)
        return self.clip_C * torch.tanh(scores)


# ─────────────────────────────────────────────────────────────────────────────
# Decoder principal
# ─────────────────────────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """
    Pointer decoder completo para el agente AM-Actor-Critic.

    Parámetros
    ----------
    d_h            : int   — dimensión del embedding (igual que AttentionEncoder).
    n_heads_glimpse: int   — cabezas para el glimpse (default 8).
    clip_C         : float — constante de clipping tanh (default 10.0).

    Uso en entrenamiento (PPO)
    --------------------------
    action, log_prob, entropy = decoder.act(context, embeddings, mask)

    Uso en evaluación (greedy)
    --------------------------
    action = decoder.greedy_action(context, embeddings, mask)

    Recálculo para actualización PPO
    ---------------------------------
    log_prob, entropy = decoder.evaluate_action(context, embeddings, mask, action)
    """

    def __init__(
        self,
        d_h:             int,
        n_heads_glimpse: int   = 8,
        clip_C:          float = 10.0,
    ):
        super().__init__()
        self.glimpse = GlimpseAttention(d_h, n_heads_glimpse)
        self.pointer = PointerAttention(d_h, clip_C)

    # ── Auxiliar privado ──────────────────────────────────────────────────────

    def _logits(
        self,
        context:    torch.Tensor,   # (batch, d_h)
        embeddings: torch.Tensor,   # (batch, N, d_h)
        mask:       torch.Tensor,   # (batch, N) bool — True = inválido
    ) -> torch.Tensor:
        """
        Calcula los logits enmascarados (batch, N).

        Pasos:
          1. Glimpse: refina el contexto usando solo los nodos válidos.
          2. Pointer: calcula scores de compatibilidad con clipping tanh.
          3. Aplica la máscara: logits de nodos inválidos → -inf.
        """
        h_refined = self.glimpse(context, embeddings, mask)    # (batch, d_h)
        logits    = self.pointer(h_refined, embeddings)         # (batch, N)
        check_tensor("decoder raw logits", logits)
        logits    = logits.masked_fill(mask, float("-inf"))
        # Verify no NaN crept in at valid positions after masking
        if torch.isnan(logits[~mask]).any():
            raise RuntimeError("decoder masked logits: NaN at valid positions after masking")
        return logits

    @staticmethod
    def _bool_mask(action_mask: torch.Tensor) -> torch.Tensor:
        """Convierte la máscara Gymnasium (1=válido) a bool (True=inválido)."""
        return action_mask == 0

    # ── API pública ───────────────────────────────────────────────────────────

    def act(
        self,
        context:     torch.Tensor,   # (batch, d_h)
        embeddings:  torch.Tensor,   # (batch, N, d_h)
        action_mask: torch.Tensor,   # (batch, N) int8: 1=válido, 0=inválido
    ):
        """
        Samplea una acción durante el entrenamiento PPO.

        Retorna
        -------
        action   : Tensor (batch,) int64 — índice del nodo seleccionado.
        log_prob : Tensor (batch,)        — log π(a|s).
        entropy  : Tensor (batch,)        — entropía de la distribución.
        """
        mask   = self._bool_mask(action_mask)
        logits = self._logits(context, embeddings, mask)
        probs  = F.softmax(logits, dim=-1)
        check_tensor("decoder softmax probs", probs)
        probs  = torch.clamp(probs, min=1e-8)
        dist   = Categorical(probs=probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def greedy_action(
        self,
        context:     torch.Tensor,   # (batch, d_h)
        embeddings:  torch.Tensor,   # (batch, N, d_h)
        action_mask: torch.Tensor,   # (batch, N) int8
    ) -> torch.Tensor:
        """
        Selecciona la acción greedy (argmax) para evaluación.

        Retorna
        -------
        Tensor (batch,) int64 — índice del nodo con mayor logit válido.
        """
        mask   = self._bool_mask(action_mask)
        valid_actions = action_mask.sum(dim=-1)
        if (valid_actions == 0).any():
            raise RuntimeError("AttentionDecoder.greedy_action: zero valid actions in mask")
        logits = self._logits(context, embeddings, mask)
        return logits.argmax(dim=-1)

    def evaluate_action(
        self,
        context:     torch.Tensor,   # (batch, d_h)
        embeddings:  torch.Tensor,   # (batch, N, d_h)
        action_mask: torch.Tensor,   # (batch, N) int8
        action:      torch.Tensor,   # (batch,) int64
    ):
        """
        Recalcula log_prob y entropía para un batch almacenado.
        Usado en la fase de actualización del loop PPO.

        Retorna
        -------
        log_prob : Tensor (batch,)
        entropy  : Tensor (batch,)
        """
        mask   = self._bool_mask(action_mask)
        logits = self._logits(context, embeddings, mask)
        probs  = F.softmax(logits, dim=-1)
        check_tensor("decoder softmax probs (eval)", probs)
        probs  = torch.clamp(probs, min=1e-8)
        dist   = Categorical(probs=probs)
        return dist.log_prob(action), dist.entropy()

    def forward(
        self,
        context:     torch.Tensor,
        embeddings:  torch.Tensor,
        action_mask: torch.Tensor,
    ):
        """
        Alias de act() para uso como nn.Module estándar en entrenamiento.
        """
        return self.act(context, embeddings, action_mask)
