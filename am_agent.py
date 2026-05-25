"""
am_agent.py
===========
Agente AM-Actor-Critic mínimo para ruteo.

Combina AttentionEncoder + ContextNetwork + AttentionDecoder en un único
nn.Module. Expone:

  generate_route()   — rollout greedy (epsilon=0) para evaluación o validación.
  act()              — paso estocástico para entrenamiento PPO.
  act_with_value()   — paso estocástico con estimación de valor (recolección PPO).
  estimate_value()   — estimación V(s) para bootstrap en episodios truncados.

El techo de pasos en generate_route() y en _encode_step() es num_nodes
(máximo de arcos posibles en un ciclo hamiltoniano), no un conteo arbitrario.
"""

import numpy as np
import torch
import torch.nn as nn

from config import MAX_DURATION
from attention_encoder import (
    AttentionEncoder,
    ContextNetwork,
    build_node_features,
    build_temporal_features,
    N_NODE_FEATURES,
)
from attention_decoder import AttentionDecoder


class AMRoutingAgent(nn.Module):
    """
    Agente de ruteo basado en Attention Model.

    Parámetros
    ----------
    num_nodes : int   — tamaño del grafo (escalable).
    d_h       : int   — dimensión del embedding (default 128).
    n_heads   : int   — cabezas de atención (default 8, debe dividir d_h).
    n_layers  : int   — capas del encoder Transformer (default 3).
    d_ff      : int   — dimensión interna FFN (default 512).
    clip_C    : float — clipping tanh del pointer (default 10.0).
    device    : torch.device
    """

    def __init__(
        self,
        num_nodes: int,
        d_h:       int   = 128,
        n_heads:   int   = 8,
        n_layers:  int   = 3,
        d_ff:      int   = 512,
        clip_C:    float = 10.0,
        device:    torch.device = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_h       = d_h
        self.device    = device or torch.device("cpu")

        self.encoder     = AttentionEncoder(N_NODE_FEATURES, d_h, n_heads, n_layers, d_ff)
        self.context_net = ContextNetwork(d_h, n_market=0)
        self.decoder     = AttentionDecoder(d_h, n_heads_glimpse=n_heads, clip_C=clip_C)

        self.to(self.device)

    # ── Auxiliares ────────────────────────────────────────────────────────────

    def _encode_step(
        self,
        current_node:           int,
        start_node:             int,
        visited_set:            set,
        time_elapsed:           float,
        step_count:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
    ):
        """
        Construye tensores de entrada y ejecuta encoder + context_net.

        Retorna
        -------
        embeddings : Tensor (1, N, d_h)
        h_t        : Tensor (1, d_h)
        node_feats : np.ndarray (N, 5)
        temporal   : np.ndarray (3,)
        """
        node_feats = build_node_features(
            current_node, start_node, visited_set,
            reward_matrix_penalized, time_matrix, distance_arr,
            self.num_nodes, max_duration,
        )
        temporal = build_temporal_features(
            time_elapsed, step_count, max_duration, self.num_nodes
        )

        feats_t    = torch.from_numpy(node_feats).unsqueeze(0).to(self.device)   # (1,N,5)
        temporal_t = torch.from_numpy(temporal).unsqueeze(0).to(self.device)     # (1,3)

        embeddings, graph_emb = self.encoder(feats_t)               # (1,N,d_h), (1,d_h)
        current_emb = embeddings[:, current_node, :]                # (1, d_h)
        h_t = self.context_net(graph_emb, current_emb, temporal_t)  # (1, d_h)

        return embeddings, h_t, node_feats, temporal

    @staticmethod
    def _build_mask(
        current_node: int,
        start_node:   int,
        visited_set:  set,
        num_nodes:    int,
    ) -> torch.Tensor:
        """
        Máscara de acciones compatible con RoutingEnv._get_action_mask():
          1 = válido, 0 = inválido.
        Retorna Tensor (1, N) int8.
        """
        mask = np.ones(num_nodes, dtype=np.int8)
        mask[current_node] = 0
        for v in visited_set:
            if v != start_node:
                mask[v] = 0
        return torch.from_numpy(mask).unsqueeze(0)

    # ── API pública ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_route(
        self,
        start_node:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float = MAX_DURATION,
    ):
        """
        Rollout greedy (sin gradientes) con el modelo actual.

        El techo de pasos es self.num_nodes — techo defensivo contra bucles
        infinitos; la terminación natural ocurre al regresar al depot o cuando
        ningún nodo intermedio es factible temporalmente.

        Retorna
        -------
        route        : list[int] | None
        total_reward : float
        time_elapsed : float
        """
        self.eval()

        has_iloc   = hasattr(reward_matrix_penalized, "iloc")
        has_iloc_t = hasattr(time_matrix, "iloc")

        current_node  = start_node
        time_elapsed  = 0.0
        visited_set   = {start_node}
        visited_inter = set()
        route         = [start_node]
        total_reward  = 0.0
        returned_home = False

        for step in range(self.num_nodes):
            embeddings, h_t, _, _ = self._encode_step(
                current_node, start_node, visited_set,
                time_elapsed, step,
                reward_matrix_penalized, time_matrix, distance_arr,
                max_duration,
            )

            # Máscara base: self-loop + intermedios visitados
            mask = self._build_mask(
                current_node, start_node, visited_set, self.num_nodes
            ).to(self.device)

            # Lookahead temporal: excluir intermedios que impidan el retorno
            mask_np = mask.cpu().numpy()[0]
            for j in range(self.num_nodes):
                if mask_np[j] == 1 and j != start_node:
                    t_to_j = (
                        float(time_matrix.iloc[current_node, j])
                        if has_iloc_t else float(time_matrix[current_node][j])
                    )
                    t_j_start = (
                        float(time_matrix.iloc[j, start_node])
                        if has_iloc_t else float(time_matrix[j][start_node])
                    )
                    if time_elapsed + t_to_j + t_j_start > max_duration + 1e-6:
                        mask_np[j] = 0
            mask = torch.from_numpy(mask_np).unsqueeze(0).to(self.device)

            # Si todos los intermedios están bloqueados → forzar retorno
            valid_non_start = [j for j in range(self.num_nodes)
                               if mask_np[j] == 1 and j != start_node]
            if not valid_non_start and current_node != start_node:
                next_node = start_node
            else:
                next_node = int(self.decoder.greedy_action(h_t, embeddings, mask))

            step_time = (
                float(time_matrix.iloc[current_node, next_node])
                if has_iloc_t else float(time_matrix[current_node][next_node])
            )
            step_reward = (
                float(reward_matrix_penalized.iloc[current_node, next_node])
                if has_iloc else float(reward_matrix_penalized[current_node][next_node])
            )

            if time_elapsed + step_time > max_duration + 1e-6 and next_node != start_node:
                break

            time_elapsed  += step_time
            total_reward  += step_reward
            current_node   = next_node
            route.append(current_node)

            if current_node != start_node:
                visited_inter.add(current_node)
            visited_set.add(current_node)

            if current_node == start_node:
                returned_home = True
                break

        # Intento de retorno forzado si no cerró el ciclo
        if not returned_home and current_node != start_node:
            t_ret = (
                float(time_matrix.iloc[current_node, start_node])
                if has_iloc_t else float(time_matrix[current_node][start_node])
            )
            r_ret = (
                float(reward_matrix_penalized.iloc[current_node, start_node])
                if has_iloc else float(reward_matrix_penalized[current_node][start_node])
            )
            if time_elapsed + t_ret <= max_duration + 1e-6:
                time_elapsed += t_ret
                total_reward += r_ret
                route.append(start_node)
                returned_home = True

        self.train()

        is_valid = (
            returned_home
            and len(route) > 1
            and route[0] == start_node
            and route[-1] == start_node
        )
        if not is_valid:
            return None, -np.inf, np.inf

        return route, total_reward, time_elapsed

    def act(
        self,
        current_node:           int,
        start_node:             int,
        visited_set:            set,
        time_elapsed:           float,
        step_count:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
        action_mask:            np.ndarray,   # (N,) int8 de RoutingEnv
    ):
        """
        Paso estocástico para entrenamiento PPO.

        Retorna
        -------
        action   : int
        log_prob : Tensor escalar
        entropy  : Tensor escalar
        """
        embeddings, h_t, _, _ = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration,
        )
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
        action_t, log_prob, entropy = self.decoder.act(h_t, embeddings, mask_t)
        return int(action_t.item()), log_prob.squeeze(0), entropy.squeeze(0)

    def act_with_value(
        self,
        critic:                 nn.Module,
        current_node:           int,
        start_node:             int,
        visited_set:            set,
        time_elapsed:           float,
        step_count:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
        action_mask:            np.ndarray,   # (N,) int8 de RoutingEnv
    ):
        """
        Paso estocástico para recolección PPO: samplea acción y estima valor.

        Retorna
        -------
        action     : int
        log_prob   : Tensor escalar
        entropy    : Tensor escalar
        value      : float
        node_feats : np.ndarray (N, 5)
        temporal   : np.ndarray (3,)
        """
        embeddings, h_t, node_feats, temporal = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration,
        )
        value  = critic(h_t).item()
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
        action_t, log_prob, entropy = self.decoder.act(h_t, embeddings, mask_t)
        return int(action_t.item()), log_prob.squeeze(0), entropy.squeeze(0), value, node_feats, temporal

    def estimate_value(
        self,
        critic:                 nn.Module,
        current_node:           int,
        start_node:             int,
        visited_set:            set,
        time_elapsed:           float,
        step_count:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
    ) -> float:
        """Estima V(s) para bootstrap en episodios truncados."""
        _, h_t, _, _ = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration,
        )
        return critic(h_t).item()
