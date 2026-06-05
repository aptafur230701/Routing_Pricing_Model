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

    def _beam_search(
        self,
        start_node:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float,
        beam_width:             int,
    ):
        """
        Rollout unificado con beam search para cualquier beam_width >= 1.

        Con beam_width=1 produce exactamente el mismo resultado que el greedy
        original (el nodo de mayor logit equivale a greedy_action). Con
        beam_width>1 mantiene los k mejores caminos parciales ordenados por
        total_reward acumulado.

        El techo de pasos es self.num_nodes — techo defensivo contra bucles
        infinitos; la terminación natural ocurre al regresar al depot o cuando
        ningún nodo intermedio es factible temporalmente.

        Retorna
        -------
        route        : list[int] | None
        total_reward : float
        time_elapsed : float
        """
        has_iloc   = hasattr(reward_matrix_penalized, "iloc")
        has_iloc_t = hasattr(time_matrix, "iloc")

        beams = [{
            "current_node":  start_node,
            "time_elapsed":  0.0,
            "visited_set":   {start_node},
            "visited_inter": set(),
            "route":         [start_node],
            "total_reward":  0.0,
            "returned_home": False,
        }]

        for step in range(self.num_nodes):
            next_beams = []

            for beam in beams:
                if beam["returned_home"]:
                    next_beams.append(beam)
                    continue

                current_node = beam["current_node"]
                time_elapsed = beam["time_elapsed"]
                visited_set  = set(beam["visited_set"])

                embeddings, h_t, _, _ = self._encode_step(
                    current_node, start_node, visited_set,
                    time_elapsed, step,
                    reward_matrix_penalized, time_matrix, distance_arr,
                    max_duration,
                )

                # Máscara base: self-loop + intermedios visitados
                mask_int = self._build_mask(
                    current_node, start_node, visited_set, self.num_nodes
                ).to(self.device)

                # Lookahead temporal: excluir intermedios que impidan el retorno
                mask_np = mask_int.cpu().numpy()[0]
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

                valid_non_start = [j for j in range(self.num_nodes)
                                   if mask_np[j] == 1 and j != start_node]

                if not valid_non_start and current_node != start_node:
                    # Forzar retorno al depot como único candidato
                    candidate_nodes = [start_node]
                else:
                    # Obtener logits del decoder sin samplear
                    bool_mask = (mask_int == 0)   # True = inválido, (1, N)
                    logits = self.decoder._logits(h_t, embeddings, bool_mask)  # (1, N)
                    logits_np = logits[0].cpu().numpy()

                    valid_nodes = [j for j in range(self.num_nodes)
                                   if mask_np[j] == 1]
                    if not valid_nodes:
                        next_beams.append(beam)
                        continue

                    # Top-k nodos válidos por logit
                    valid_nodes.sort(key=lambda j: logits_np[j], reverse=True)
                    candidate_nodes = valid_nodes[:beam_width]

                # Crear beams hijos para cada candidato
                for next_node in candidate_nodes:
                    step_time = (
                        float(time_matrix.iloc[current_node, next_node])
                        if has_iloc_t else float(time_matrix[current_node][next_node])
                    )
                    step_reward = (
                        float(reward_matrix_penalized.iloc[current_node, next_node])
                        if has_iloc else float(reward_matrix_penalized[current_node][next_node])
                    )

                    if time_elapsed + step_time > max_duration + 1e-6 and next_node != start_node:
                        continue

                    new_visited_set   = set(visited_set)
                    new_visited_inter = set(beam["visited_inter"])
                    new_visited_set.add(next_node)
                    if next_node != start_node:
                        new_visited_inter.add(next_node)

                    next_beams.append({
                        "current_node":  next_node,
                        "time_elapsed":  time_elapsed + step_time,
                        "visited_set":   new_visited_set,
                        "visited_inter": new_visited_inter,
                        "route":         beam["route"] + [next_node],
                        "total_reward":  beam["total_reward"] + step_reward,
                        "returned_home": next_node == start_node,
                    })

            if not next_beams:
                break

            next_beams.sort(key=lambda b: b["total_reward"], reverse=True)
            beams = next_beams[:beam_width]

            if all(b["returned_home"] for b in beams):
                break

        # Intento de retorno forzado para beams que no cerraron el ciclo
        for beam in beams:
            if not beam["returned_home"] and beam["current_node"] != start_node:
                cn = beam["current_node"]
                t_ret = (
                    float(time_matrix.iloc[cn, start_node])
                    if has_iloc_t else float(time_matrix[cn][start_node])
                )
                r_ret = (
                    float(reward_matrix_penalized.iloc[cn, start_node])
                    if has_iloc else float(reward_matrix_penalized[cn][start_node])
                )
                if beam["time_elapsed"] + t_ret <= max_duration + 1e-6:
                    beam["time_elapsed"] += t_ret
                    beam["total_reward"] += r_ret
                    beam["route"].append(start_node)
                    beam["returned_home"] = True

        best_beam   = None
        best_reward = -np.inf
        for beam in beams:
            is_valid = (
                beam["returned_home"]
                and len(beam["route"]) > 1
                and beam["route"][0] == start_node
                and beam["route"][-1] == start_node
            )
            if is_valid and beam["total_reward"] > best_reward:
                best_reward = beam["total_reward"]
                best_beam   = beam

        if best_beam is None:
            return None, -np.inf, np.inf

        return best_beam["route"], best_beam["total_reward"], best_beam["time_elapsed"]

    # ── API pública ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_route(
        self,
        start_node:             int,
        reward_matrix_penalized,
        time_matrix,
        distance_arr:           np.ndarray,
        max_duration:           float = MAX_DURATION,
        beam_width:             int   = None,
    ):
        """
        Rollout determinista (sin gradientes) con el modelo actual.

        Delega la lógica de búsqueda a _beam_search(), que funciona para
        cualquier beam_width >= 1 (beam_width=1 produce el mismo resultado
        que el greedy original).

        Cuando beam_width es None, lo resuelve automáticamente vía
        get_beam_width(self.num_nodes): modelos pequeños (≤10 nodos) usan
        beam=5, tamaños intermedios (≤35) usan beam=3, y modelos grandes
        usan beam=1 (política robusta, greedy suficiente).

        El techo de pasos en _beam_search() es self.num_nodes — techo
        defensivo contra bucles infinitos; la terminación natural ocurre al
        regresar al depot o cuando ningún nodo intermedio es factible
        temporalmente.

        Retorna
        -------
        route        : list[int] | None
        total_reward : float
        time_elapsed : float
        """
        from config import get_beam_width as _get_beam_width
        if beam_width is None:
            beam_width = _get_beam_width(self.num_nodes)

        self.eval()
        try:
            return self._beam_search(
                start_node, reward_matrix_penalized, time_matrix,
                distance_arr, max_duration, beam_width,
            )
        finally:
            self.train()

    @torch.no_grad()
    def beam_search_dynamic(
        self,
        start_node:    int,
        start_day_idx: int,
        rm_pen_start,           # pd.DataFrame — día de inicio, solo para features
        time_matrix,
        rate_stack:    np.ndarray,
        loads_stack:   np.ndarray,
        distance_arr:  np.ndarray,
        diesel_arr:    np.ndarray,
        max_duration:  float = MAX_DURATION,
        beam_width:    int   = None,
    ):
        """Beam search con días de mercado dinámicos por beam.

        Features del agente : rm_pen_start (día de inicio, igual que training).
        Recompensa acumulada: matriz del día corriente según time_elapsed de cada beam,
                              donde day_idx = start_day_idx + int(time_elapsed // 14).
        """
        from config import get_beam_width as _get_beam_width
        from problem_data import build_day_matrices

        if beam_width is None:
            beam_width = _get_beam_width(self.num_nodes)

        has_iloc_t = hasattr(time_matrix, "iloc")
        max_day    = rate_stack.shape[0] - 1
        day_cache  = {}  # day_idx → rm_penalized; evita reconstruir el mismo día

        def get_rm(day_idx):
            if day_idx not in day_cache:
                _, rm = build_day_matrices(
                    rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
                )
                day_cache[day_idx] = rm
            return day_cache[day_idx]

        beams = [{
            "current_node":  start_node,
            "time_elapsed":  0.0,
            "visited_set":   {start_node},
            "visited_inter": set(),
            "route":         [start_node],
            "total_reward":  0.0,
            "returned_home": False,
        }]

        for step in range(self.num_nodes):
            next_beams = []

            for beam in beams:
                if beam["returned_home"]:
                    next_beams.append(beam)
                    continue

                current_node = beam["current_node"]
                time_elapsed = beam["time_elapsed"]
                visited_set  = set(beam["visited_set"])

                day_offset = int(time_elapsed // 14)
                rm_day     = get_rm(min(start_day_idx + day_offset, max_day))

                embeddings, h_t, _, _ = self._encode_step(
                    current_node, start_node, visited_set,
                    time_elapsed, step,
                    rm_pen_start, time_matrix, distance_arr,
                    max_duration,
                )

                mask_int = self._build_mask(
                    current_node, start_node, visited_set, self.num_nodes
                ).to(self.device)

                mask_np = mask_int.cpu().numpy()[0]
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

                valid_non_start = [j for j in range(self.num_nodes)
                                   if mask_np[j] == 1 and j != start_node]

                if not valid_non_start and current_node != start_node:
                    candidate_nodes = [start_node]
                else:
                    bool_mask = (mask_int == 0)
                    logits    = self.decoder._logits(h_t, embeddings, bool_mask)
                    logits_np = logits[0].cpu().numpy()

                    valid_nodes = [j for j in range(self.num_nodes) if mask_np[j] == 1]
                    if not valid_nodes:
                        next_beams.append(beam)
                        continue

                    valid_nodes.sort(key=lambda j: logits_np[j], reverse=True)
                    candidate_nodes = valid_nodes[:beam_width]

                for next_node in candidate_nodes:
                    step_time = (
                        float(time_matrix.iloc[current_node, next_node])
                        if has_iloc_t else float(time_matrix[current_node][next_node])
                    )
                    if time_elapsed + step_time > max_duration + 1e-6 and next_node != start_node:
                        continue

                    step_reward = float(rm_day.iloc[current_node, next_node])

                    new_visited_set   = set(visited_set)
                    new_visited_inter = set(beam["visited_inter"])
                    new_visited_set.add(next_node)
                    if next_node != start_node:
                        new_visited_inter.add(next_node)

                    next_beams.append({
                        "current_node":  next_node,
                        "time_elapsed":  time_elapsed + step_time,
                        "visited_set":   new_visited_set,
                        "visited_inter": new_visited_inter,
                        "route":         beam["route"] + [next_node],
                        "total_reward":  beam["total_reward"] + step_reward,
                        "returned_home": next_node == start_node,
                    })

            if not next_beams:
                break

            next_beams.sort(key=lambda b: b["total_reward"], reverse=True)
            beams = next_beams[:beam_width]

            if all(b["returned_home"] for b in beams):
                break

        for beam in beams:
            if not beam["returned_home"] and beam["current_node"] != start_node:
                cn         = beam["current_node"]
                day_offset = int(beam["time_elapsed"] // 14)
                rm_day     = get_rm(min(start_day_idx + day_offset, max_day))
                t_ret = (
                    float(time_matrix.iloc[cn, start_node])
                    if has_iloc_t else float(time_matrix[cn][start_node])
                )
                r_ret = float(rm_day.iloc[cn, start_node])
                if beam["time_elapsed"] + t_ret <= max_duration + 1e-6:
                    beam["time_elapsed"] += t_ret
                    beam["total_reward"] += r_ret
                    beam["route"].append(start_node)
                    beam["returned_home"] = True

        best_beam   = None
        best_reward = -np.inf
        for beam in beams:
            is_valid = (
                beam["returned_home"]
                and len(beam["route"]) > 1
                and beam["route"][0] == start_node
                and beam["route"][-1] == start_node
            )
            if is_valid and beam["total_reward"] > best_reward:
                best_reward = beam["total_reward"]
                best_beam   = beam

        if best_beam is None:
            return None, -np.inf, np.inf

        return best_beam["route"], best_beam["total_reward"], best_beam["time_elapsed"]

    @torch.no_grad()
    def greedy_action(
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
    ) -> int:
        """Selección determinista (argmax de logits) para evaluación env-based."""
        embeddings, h_t, _, _ = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration,
        )
        mask_t    = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
        bool_mask = (mask_t == 0)
        logits    = self.decoder._logits(h_t, embeddings, bool_mask)  # (1, N)
        return int(logits[0].argmax().item())

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
