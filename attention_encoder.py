"""
attention_encoder.py
====================
Transformer encoder for the AM-Actor-Critic routing agent.

Componentes:
  NodeFeatureProjection  — proyecta features crudas de nodos a dimensión d_h
  MultiHeadSelfAttention — atención multi-cabeza estándar (scaled dot-product)
  TransformerEncoderLayer — MHA + FFN + LayerNorm con residual
  AttentionEncoder        — apila L capas, produce embeddings por nodo
                            y embedding global del grafo
  ContextNetwork          — combina embedding global + nodo actual + features
                            temporales → vector de contexto h_t

El vector de contexto h_t es compartido por:
  · AttentionDecoder  (decisión de ruteo)
  · CriticHead        (función de valor V(s))
  · PricingHead       (precio de reserva p*)

Features por nodo j (desde la posición actual del camión):
  [0] reward_norm   — ganancia neta del arco actual→j (día actual), normalizada
  [1] time_norm     — tiempo de viaje actual→j, normalizado por max_duration
  [2] dist_norm     — distancia actual→j, normalizada por máximo
  [3] is_depot      — 1.0 si j es el nodo de inicio del episodio, 0.0 si no
  [4] visited       — 1.0 si j ya fue visitado en el episodio, 0.0 si no
  [5] trucks_norm   — camiones disponibles en j para el delta correspondiente
  [6] avail_prior   — P(lane actual→j existe) según historial; prior del Bernoulli
  [7] reward_abs    — reward crudo / REWARD_GLOBAL_P95, clipeado a [0, 1]; 0 si negativo

Compatibilidad:
  · Sin dependencias de agent.py, training.py, routing_env.py ni evaluation.py.
  · routing_env.py permanece sin cambios.
  · Las matrices (reward, time, distance) aceptan pd.DataFrame o np.ndarray.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LTR_CLIP, TRUCKS_CLIP

# Número de features por nodo — debe coincidir con build_node_features()
N_NODE_FEATURES = 8


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: construcción de la matriz de features de nodos
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features(
    current_node:           int,
    start_node:             int,
    visited_set:            set,
    reward_matrix_penalized: np.ndarray,  # (N×N)
    time_matrix:             np.ndarray,  # (N×N)
    distance_arr:            np.ndarray,  # (N×N)
    num_nodes:               int,
    max_duration:            float,
    trucks_stack:            np.ndarray = None,  # (N, 3) — camiones por nodo×delta
    time_matrix_arr:         np.ndarray = None,  # unused; kept for API compatibility
    avail_prob_arr:          np.ndarray = None,  # (N, N) — prior Bernoulli por arco
    reward_global_p95:       float      = 1.0,
) -> np.ndarray:
    """
    Construye la matriz de features de nodos para el encoder (vectorizada).

    Cada fila j contiene las features del nodo j vistas desde current_node:
      [reward_norm, time_norm, dist_norm, is_depot, visited, trucks_norm,
       avail_prior, reward_abs]

    La diagonal de reward_matrix_penalized tiene BIG_M_PENALTY (-1e9).
    Se recorta a [-1e4, 1e4] antes de normalizar para evitar desbordamientos.

    Retorna np.ndarray de forma (N, N_NODE_FEATURES), dtype float32.
    """
    feats = np.zeros((num_nodes, N_NODE_FEATURES), dtype=np.float32)

    # Columns 0-2: reward / time / distance, all from current_node row
    feats[:, 0] = reward_matrix_penalized[current_node]
    feats[:, 1] = time_matrix[current_node]
    feats[:, 2] = distance_arr[current_node]

    # Column 3: is_depot
    feats[start_node, 3] = 1.0

    # Column 4: visited (vectorized — includes start_node since it's always in visited_set)
    if visited_set:
        feats[list(visited_set), 4] = 1.0

    # Column 5: trucks_norm (vectorized delta_idx via fancy indexing)
    if trucks_stack is not None:
        tm = time_matrix_arr if time_matrix_arr is not None else time_matrix
        travel_hours = tm[current_node]                                       # (N,)
        delta_idx = np.clip(
            np.ceil(travel_hours / 14.0).astype(np.int32) - 1, 0, 2
        )                                                                      # (N,) int32
        trucks_vals = trucks_stack[np.arange(num_nodes), delta_idx]           # (N,)
        feats[:, 5] = np.minimum(trucks_vals, TRUCKS_CLIP) / TRUCKS_CLIP

    # Column 6: avail_prior
    if avail_prob_arr is not None:
        feats[:, 6] = avail_prob_arr[current_node]

    # Recortar reward para evitar BIG_M_PENALTY (-1e9) en la diagonal
    feats[:, 0] = np.clip(feats[:, 0], -1e4, 1e4)
    # Copia antes de normalizar: preserva la escala absoluta para feature [7]
    raw_rewards = feats[:, 0].copy()

    # reward y distancia:
    # normalización relativa intra-fila para preservar ranking local
    # entre las alternativas disponibles desde current_node.
    # Esto ayuda al encoder a comparar qué nodos son mejores/peores
    # dentro del estado actual independientemente de la escala absoluta.
    for col in (0, 2):
        col_min = feats[:, col].min()
        col_max = feats[:, col].max()
        if col_max > col_min:
            feats[:, col] = (feats[:, col] - col_min) / (col_max - col_min)
        else:
            feats[:, col] = 0.0

    # tiempo:
    # normalización absoluta respecto al presupuesto global
    # para preservar información de factibilidad temporal.
    # Un arco largo debe seguir viéndose "caro" temporalmente aunque
    # todos los demás arcos también sean largos.
    # Se aplica UNA sola vez para preservar la escala temporal real.
    feats[:, 1] = np.clip(feats[:, 1] / max(max_duration, 1e-6), 0.0, 1.0)

    # Feature [7]: reward absoluto normalizado por el P95 global del entrenamiento.
    # Complementa la feature [0] (ranking local) con información de escala absoluta.
    # Valores negativos (arcos con costo neto) quedan en 0.0.
    _denom = max(reward_global_p95, 1e-6)
    feat7 = np.clip(raw_rewards / _denom, 0.0, 1.0)
    feat7[raw_rewards < 0] = 0.0
    feats[:, 7] = feat7

    if np.isnan(feats).any():
        bad = np.argwhere(np.isnan(feats)).tolist()
        raise RuntimeError(
            f"build_node_features: NaN in feature matrix "
            f"(current_node={current_node}, positions={bad})"
        )
    if np.isinf(feats).any():
        bad = np.argwhere(np.isinf(feats)).tolist()
        raise RuntimeError(
            f"build_node_features: Inf in feature matrix "
            f"(current_node={current_node}, positions={bad})"
        )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Bloques del Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

class NodeFeatureProjection(nn.Module):
    """
    Proyección lineal de las features crudas de nodos a la dimensión de embedding.

    Entrada : (batch, N, d_input)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_input: int, d_h: int):
        super().__init__()
        self.proj = nn.Linear(d_input, d_h)
        self.norm = nn.LayerNorm(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(F.relu(self.proj(x)))


class MultiHeadSelfAttention(nn.Module):
    """
    Atención multi-cabeza estándar (scaled dot-product).

    Entrada : (batch, N, d_h)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_h: int, n_heads: int):
        super().__init__()
        assert d_h % n_heads == 0, "d_h debe ser divisible por n_heads"
        self.n_heads = n_heads
        self.d_k = d_h // n_heads

        self.W_q = nn.Linear(d_h, d_h, bias=False)
        self.W_k = nn.Linear(d_h, d_h, bias=False)
        self.W_v = nn.Linear(d_h, d_h, bias=False)
        self.W_o = nn.Linear(d_h, d_h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        def split_heads(t):
            return t.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        Q = split_heads(self.W_q(x))   # (B, H, N, d_k)
        K = split_heads(self.W_k(x))
        V = split_heads(self.W_v(x))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn   = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)                         # (B, H, N, d_k)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)  # (B, N, d_h)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """
    Una capa del encoder Transformer: MHA + FFN con residual y LayerNorm.

    Entrada : (batch, N, d_h)
    Salida  : (batch, N, d_h)
    """

    def __init__(self, d_h: int, n_heads: int, d_ff: int):
        super().__init__()
        self.mha   = MultiHeadSelfAttention(d_h, n_heads)
        self.norm1 = nn.LayerNorm(d_h)
        self.ffn   = nn.Sequential(
            nn.Linear(d_h, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_h),
        )
        self.norm2 = nn.LayerNorm(d_h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ffn(x))
        return x


class AttentionEncoder(nn.Module):
    """
    Encoder Transformer: mapea N vectores de features a N embeddings ricos.

    Parámetros
    ----------
    d_input  : int — features por nodo (N_NODE_FEATURES = 8).
    d_h      : int — dimensión del embedding (ej. 128).
    n_heads  : int — cabezas de atención (ej. 8, debe dividir d_h).
    n_layers : int — capas del encoder (ej. 3).
    d_ff     : int — dimensión interna del FFN (ej. 512 = 4 × d_h).

    Forward
    -------
    node_features : Tensor (batch, N, d_input)

    Retorna
    -------
    embeddings      : Tensor (batch, N, d_h)  — embedding por nodo.
    graph_embedding : Tensor (batch, d_h)     — promedio de todos los embeddings.
    """

    def __init__(
        self,
        d_input:  int = N_NODE_FEATURES,
        d_h:      int = 128,
        n_heads:  int = 8,
        n_layers: int = 3,
        d_ff:     int = 512,
    ):
        super().__init__()
        self.input_proj = NodeFeatureProjection(d_input, d_h)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_h, n_heads, d_ff)
            for _ in range(n_layers)
        ])

    def forward(self, node_features: torch.Tensor):
        x = self.input_proj(node_features)   # (batch, N, d_h)
        for layer in self.layers:
            x = layer(x)                     # (batch, N, d_h)
        graph_emb = x.mean(dim=1)            # (batch, d_h)
        return x, graph_emb


# ─────────────────────────────────────────────────────────────────────────────
# Vector de contexto h_t
# ─────────────────────────────────────────────────────────────────────────────

class ContextNetwork(nn.Module):
    """
    Construye el vector de contexto h_t usado por decoder, critic y pricing.

    Combina:
      · graph_embedding   (d_h)       — estado global del grafo
      · current_node_emb  (d_h)       — posición actual del camión
      · temporal_features (3)         — tiempo_norm, pasos_restantes_norm,
                                        progreso_norm
      · market_features   (n_market)  — MTI, dispersión, forecast (opcional,
                                        por defecto 0 = no se usa)

    Entrada total: 2·d_h + 3 + n_market  →  MLP  →  d_h

    Parámetros
    ----------
    d_h      : int — dimensión del embedding (debe coincidir con AttentionEncoder).
    n_market : int — número de features de mercado adicionales (default 0).
    """

    def __init__(self, d_h: int, n_market: int = 1):
        super().__init__()
        n_temporal = 3          # time_norm, remaining_steps_norm, progress_norm
        d_in = 2 * d_h + n_temporal + n_market
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
        )

    def forward(
        self,
        graph_emb:        torch.Tensor,   # (batch, d_h)
        current_node_emb: torch.Tensor,   # (batch, d_h)
        temporal_feats:   torch.Tensor,   # (batch, 3)
        market_feats:     torch.Tensor = None,  # (batch, n_market) | None
    ) -> torch.Tensor:
        """
        Retorna el vector de contexto h_t de forma (batch, d_h).
        """
        parts = [graph_emb, current_node_emb, temporal_feats]
        if market_feats is not None:
            parts.append(market_feats)
        h = torch.cat(parts, dim=-1)
        return self.proj(h)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: features de mercado para ContextNetwork
# ─────────────────────────────────────────────────────────────────────────────

def build_market_features(
    current_node: int,
    day_idx:      int,
    ltr_stack:    np.ndarray,   # [num_nodes, 120]
) -> np.ndarray:
    """Construye el vector de market features (1,) para ContextNetwork.

    [0] ltr_norm — LTR del hub actual en el día actual, normalizado a [0,1].
    """
    ltr_raw  = float(ltr_stack[current_node, day_idx])
    ltr_norm = min(ltr_raw, LTR_CLIP) / LTR_CLIP
    return np.array([ltr_norm], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Utilidad: features temporales desde el estado del entorno
# ─────────────────────────────────────────────────────────────────────────────

def build_node_features_batch(
    current_node:      np.ndarray,   # (B,) int64
    start_node:        np.ndarray,   # (B,) int64
    visited_mask:      np.ndarray,   # (B, N) bool
    rm_pen_stack:      np.ndarray,   # (num_train_days, N, N)
    time_matrix_arr:   np.ndarray,   # (N, N)
    distance_arr:      np.ndarray,   # (N, N)
    day_idx:           np.ndarray,   # (B,) int64
    num_nodes:         int,
    max_duration:      float,
    trucks_stack_raw:  np.ndarray = None,   # (N, num_days_ltr, 3)
    avail_prob_arr:    np.ndarray = None,   # (N, N)
    reward_global_p95: float      = 1.0,
) -> np.ndarray:                            # (B, N, N_NODE_FEATURES=8) float32
    """
    Batched version of build_node_features.

    Computes the (N, 8) feature matrix for each of B episodes simultaneously
    using numpy fancy indexing — no Python loops over N or B.

    trucks_stack_raw shape is (N, num_days_ltr, 3): the full trucks stack.
    day_idx is used to select the appropriate day slice per episode.
    """
    B = len(current_node)
    N = num_nodes
    feats = np.zeros((B, N, N_NODE_FEATURES), dtype=np.float32)

    # ── f0: reward from rm_pen_stack ──────────────────────────────────────────
    # rm_pen_stack[day_idx[b], current_node[b], j] for all j → (B, N)
    feats[:, :, 0] = rm_pen_stack[day_idx, current_node, :]   # fancy: (B, N)

    # ── f1: travel time from current node to each j ───────────────────────────
    feats[:, :, 1] = time_matrix_arr[current_node, :]          # (B, N)

    # ── f2: distance from current node to each j ─────────────────────────────
    feats[:, :, 2] = distance_arr[current_node, :]             # (B, N)

    # ── f3: is_depot — 1.0 if j == start_node[b] ─────────────────────────────
    feats[:, :, 3] = (np.arange(N)[None, :] == start_node[:, None]).astype(np.float32)

    # ── f4: visited — 1.0 if j in visited_mask[b] ────────────────────────────
    feats[:, :, 4] = visited_mask.astype(np.float32)

    # ── f5: trucks availability ───────────────────────────────────────────────
    if trucks_stack_raw is not None:
        travel_hours = time_matrix_arr[current_node, :]                 # (B, N)
        delta_idx = np.clip(
            np.ceil(travel_hours / 14.0).astype(np.int32) - 1, 0, 2
        )                                                                # (B, N)
        # trucks_stack_raw: (N, num_days_ltr, 3)
        max_ltr_day = trucks_stack_raw.shape[1] - 1
        day_idx_t = np.minimum(day_idx, max_ltr_day)
        trucks_val = trucks_stack_raw[
            np.arange(N)[None, :],    # (1, N) → (B, N)
            day_idx_t[:, None],       # (B, 1) → (B, N)
            delta_idx,                # (B, N)
        ]                                                                # (B, N)
        feats[:, :, 5] = np.minimum(trucks_val, TRUCKS_CLIP) / TRUCKS_CLIP

    # ── f6: lane availability prior ───────────────────────────────────────────
    if avail_prob_arr is not None:
        feats[:, :, 6] = avail_prob_arr[current_node, :]       # (B, N)

    # ── Clip reward before normalization (avoids BIG_M_PENALTY overflow) ─────
    feats[:, :, 0] = np.clip(feats[:, :, 0], -1e4, 1e4)
    raw_rewards = feats[:, :, 0].copy()   # (B, N) — before normalization for f7

    # ── Row-wise min-max normalization for f0 (reward) and f2 (distance) ─────
    for col in (0, 2):
        col_min = feats[:, :, col].min(axis=1, keepdims=True)   # (B, 1)
        col_max = feats[:, :, col].max(axis=1, keepdims=True)   # (B, 1)
        rng = col_max - col_min
        safe = (rng > 0).squeeze(axis=1)                         # (B,)
        feats[safe, :, col] = (
            (feats[safe, :, col] - col_min[safe]) / rng[safe]
        )
        feats[~safe, :, col] = 0.0

    # ── Absolute normalization for f1 (time) ─────────────────────────────────
    feats[:, :, 1] = np.clip(feats[:, :, 1] / max(max_duration, 1e-6), 0.0, 1.0)

    # ── f7: absolute reward normalized by global P95 ──────────────────────────
    _denom = max(reward_global_p95, 1e-6)
    feat7 = np.clip(raw_rewards / _denom, 0.0, 1.0)
    feat7[raw_rewards < 0] = 0.0
    feats[:, :, 7] = feat7

    return feats


def build_temporal_features_batch(
    time_elapsed: np.ndarray,   # (B,) float32
    step_count:   np.ndarray,   # (B,) int64
    max_duration: float,
    num_nodes:    int,
) -> np.ndarray:                # (B, 3) float32
    """Batched version of build_temporal_features."""
    denom = max(max_duration, 1e-6)
    out = np.empty((len(time_elapsed), 3), dtype=np.float32)
    out[:, 0] = np.minimum(time_elapsed, max_duration) / denom
    out[:, 1] = np.maximum(0.0, max_duration - time_elapsed) / denom
    out[:, 2] = step_count / max(num_nodes, 1)
    return out


def build_market_features_batch(
    current_node: np.ndarray,   # (B,) int64
    day_idx:      np.ndarray,   # (B,) int64
    ltr_stack:    np.ndarray,   # (N, num_ltr_days)
) -> np.ndarray:                # (B, 1) float32
    """Batched version of build_market_features."""
    max_ltr_day = ltr_stack.shape[1] - 1
    day_idx_safe = np.minimum(day_idx, max_ltr_day)
    ltr_raw = ltr_stack[current_node, day_idx_safe].astype(np.float32)   # (B,)
    ltr_norm = np.clip(ltr_raw, 0.0, LTR_CLIP) / LTR_CLIP
    return ltr_norm[:, None]   # (B, 1)


def build_temporal_features(
    time_elapsed: float,
    step_count:   int,
    max_duration: float,
    num_nodes:    int,
) -> np.ndarray:
    """
    Construye el vector de features temporales (3,) para ContextNetwork.

    [0] elapsed_time_norm   = time_elapsed / max_duration           (0-1)
    [1] remaining_time_norm = max(0, max_duration - time_elapsed) / max_duration  (0-1)
    [2] progress_fraction   = step_count / num_nodes                (0-1)

    El tensor sigue siendo shape (3,) — sin cambio de dimensión.
    """
    denom = max(max_duration, 1e-6)
    return np.array([
        min(time_elapsed, max_duration) / denom,
        max(0.0, max_duration - time_elapsed) / denom,
        step_count / max(num_nodes, 1),
    ], dtype=np.float32)
