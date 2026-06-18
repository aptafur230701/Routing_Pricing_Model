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

from config import MAX_DURATION, REWARD_SCALE_FACTOR
from attention_encoder import (
    AttentionEncoder,
    ContextNetwork,
    build_node_features,
    build_market_features,
    build_temporal_features,
    build_node_features_batch,
    build_temporal_features_batch,
    build_market_features_batch,
    N_NODE_FEATURES,
)
from attention_decoder import AttentionDecoder
from problem_data import draw_lane_availability
from debug_utils import check_tensor


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
        self.context_net = ContextNetwork(d_h, n_market=1)
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
        ltr_stack:              np.ndarray,        # [num_nodes, 120]
        trucks_stack:           np.ndarray,        # [num_nodes, 120, 3]
        day_idx:                int,
        avail_prob_arr:         np.ndarray = None, # [num_nodes, num_nodes]
        reward_global_p95:      float      = 1.0,
    ):
        """
        Construye tensores de entrada y ejecuta encoder + context_net.

        Retorna
        -------
        embeddings : Tensor (1, N, d_h)
        h_t        : Tensor (1, d_h)
        node_feats : np.ndarray (N, N_NODE_FEATURES)
        temporal   : np.ndarray (3,)
        market     : np.ndarray (1,)
        """
        trucks_day = trucks_stack[:, day_idx, :]   # (num_nodes, 3)

        node_feats = build_node_features(
            current_node, start_node, visited_set,
            reward_matrix_penalized, time_matrix, distance_arr,
            self.num_nodes, max_duration,
            trucks_stack=trucks_day,
            avail_prob_arr=avail_prob_arr,
            reward_global_p95=reward_global_p95,
        )
        temporal = build_temporal_features(
            time_elapsed, step_count, max_duration, self.num_nodes
        )
        market = build_market_features(current_node, day_idx, ltr_stack)

        feats_t    = torch.from_numpy(node_feats).unsqueeze(0).to(self.device)   # (1,N,N_NODE_FEATURES)
        temporal_t = torch.from_numpy(temporal).unsqueeze(0).to(self.device)     # (1,3)
        market_t   = torch.from_numpy(market).unsqueeze(0).to(self.device)       # (1,1)

        embeddings, graph_emb = self.encoder(feats_t)                            # (1,N,d_h), (1,d_h)
        check_tensor("_encode_step embeddings", embeddings)
        check_tensor("_encode_step graph_emb", graph_emb)
        current_emb = embeddings[:, current_node, :]                             # (1, d_h)
        h_t = self.context_net(graph_emb, current_emb, temporal_t, market_t)    # (1, d_h)
        check_tensor("_encode_step h_t", h_t)

        return embeddings, h_t, node_feats, temporal, market

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
        ltr_stack:              np.ndarray,        # [num_nodes, 120]
        trucks_stack:           np.ndarray,        # [num_nodes, 120, 3]
        day_idx:                int,
        avail_prob_arr:         np.ndarray = None, # [num_nodes, num_nodes]
        reward_global_p95:      float      = 1.0,
        critic:                 nn.Module  = None,
        value_coef:             float      = 1.0,
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

                embeddings, h_t, _, _, _ = self._encode_step(
                    current_node, start_node, visited_set,
                    time_elapsed, step,
                    reward_matrix_penalized, time_matrix, distance_arr,
                    max_duration, ltr_stack, trucks_stack, day_idx,
                    avail_prob_arr=avail_prob_arr,
                    reward_global_p95=reward_global_p95,
                )

                # Máscara base: self-loop + intermedios visitados
                mask_int = self._build_mask(
                    current_node, start_node, visited_set, self.num_nodes
                ).to(self.device)

                # Lookahead temporal (vectorizado): excluir intermedios que impidan el retorno
                mask_np    = mask_int.cpu().numpy()[0]
                non_start  = np.arange(self.num_nodes) != start_node
                t_to_j     = time_matrix[current_node]              # (N,)
                t_j_start  = time_matrix[:, start_node]             # (N,)
                over_budget = (time_elapsed + t_to_j + t_j_start) > max_duration + 1e-6
                mask_np[non_start & over_budget] = 0

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
                    step_time   = float(time_matrix[current_node, next_node])
                    step_reward = float(reward_matrix_penalized[current_node, next_node])

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

            if critic is not None:
                def _score(b):
                    if b["returned_home"]:
                        v = 0.0
                    else:
                        v = self.estimate_value(
                            critic, b["current_node"], start_node, b["visited_set"],
                            b["time_elapsed"], step + 1,
                            reward_matrix_penalized, time_matrix, distance_arr,
                            max_duration, ltr_stack, trucks_stack, day_idx,
                            avail_prob_arr=avail_prob_arr,
                            reward_global_p95=reward_global_p95,
                        )
                    return b["total_reward"] + value_coef * v * REWARD_SCALE_FACTOR

                next_beams.sort(key=_score, reverse=True)
            else:
                next_beams.sort(key=lambda b: b["total_reward"], reverse=True)
            beams = next_beams[:beam_width]

            if all(b["returned_home"] for b in beams):
                break

        # Intento de retorno forzado para beams que no cerraron el ciclo
        for beam in beams:
            if not beam["returned_home"] and beam["current_node"] != start_node:
                cn    = beam["current_node"]
                t_ret = float(time_matrix[cn, start_node])
                r_ret = float(reward_matrix_penalized[cn, start_node])
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
        max_duration:           float      = MAX_DURATION,
        beam_width:             int        = None,
        ltr_stack:              np.ndarray = None,   # [num_nodes, 120]
        trucks_stack:           np.ndarray = None,   # [num_nodes, 120, 3]
        day_idx:                int        = 0,
        avail_prob_arr:         np.ndarray = None,   # [num_nodes, num_nodes]
        reward_global_p95:      float      = 1.0,
        critic:                 nn.Module  = None,
        value_coef:             float      = 1.0,
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
                ltr_stack, trucks_stack, day_idx,
                avail_prob_arr=avail_prob_arr,
                reward_global_p95=reward_global_p95,
                critic=critic, value_coef=value_coef,
            )
        finally:
            self.train()

    @torch.no_grad()
    def beam_search_dynamic(
        self,
        start_node:        int,
        start_day_idx:     int,
        time_matrix,
        rate_stack:        np.ndarray,
        loads_stack:       np.ndarray,
        distance_arr:      np.ndarray,
        diesel_arr:        np.ndarray,
        max_duration:      float      = MAX_DURATION,
        beam_width:        int        = None,
        ltr_stack:         np.ndarray = None,   # [num_nodes, 120]
        trucks_stack:      np.ndarray = None,   # [num_nodes, 120, 3]
        avail_prob_arr:    np.ndarray = None,   # [num_nodes, num_nodes]
        reward_global_p95: float      = 1.0,
        critic:            nn.Module  = None,
        value_coef:        float      = 1.0,
    ):
        """Beam search con días de mercado dinámicos y disponibilidad estocástica.

        Features del agente: matriz del día corriente de cada beam (igual que training).
        Recompensa acumulada: misma matriz del día corriente según time_elapsed de cada beam.
        Disponibilidad estocástica: usa el MISMO esquema de semillas que RoutingEnv
          (seed basado en start_day_idx, node y arrival_day), garantizando que DRL y
          baselines compiten sobre la misma realización del mundo.
        """
        from config import get_beam_width as _get_beam_width
        from problem_data import build_day_matrices

        if beam_width is None:
            beam_width = _get_beam_width(self.num_nodes)
        max_day    = rate_stack.shape[0] - 1
        day_cache  = {}  # day_idx → rm_penalized; evita reconstruir el mismo día

        def get_rm(day_idx):
            if day_idx not in day_cache:
                _, rm = build_day_matrices(
                    rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
                )
                day_cache[day_idx] = rm
            return day_cache[day_idx]

        # Sorteo inicial: lanes desde start_node al llegar con time_elapsed=0
        if avail_prob_arr is not None:
            init_arrival_day   = min(start_day_idx, max_day)
            init_lane_exists   = draw_lane_availability(
                start_day_idx, start_node, init_arrival_day,
                avail_prob_arr, self.num_nodes,
            )
        else:
            init_lane_exists = None

        beams = [{
            "current_node":  start_node,
            "time_elapsed":  0.0,
            "visited_set":   {start_node},
            "visited_inter": set(),
            "route":         [start_node],
            "total_reward":  0.0,
            "returned_home": False,
            "lane_exists":   init_lane_exists,
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
                day_idx    = min(start_day_idx + day_offset, max_day)
                rm_day     = get_rm(day_idx)

                embeddings, h_t, _, _, _ = self._encode_step(
                    current_node, start_node, visited_set,
                    time_elapsed, step,
                    rm_day, time_matrix, distance_arr,
                    max_duration, ltr_stack, trucks_stack, day_idx,
                    avail_prob_arr=avail_prob_arr,
                    reward_global_p95=reward_global_p95,
                )

                mask_int = self._build_mask(
                    current_node, start_node, visited_set, self.num_nodes
                ).to(self.device)

                # Lookahead temporal (vectorizado)
                mask_np    = mask_int.cpu().numpy()[0]
                non_start  = np.arange(self.num_nodes) != start_node
                t_to_j     = time_matrix[current_node]
                t_j_start  = time_matrix[:, start_node]
                over_budget = (time_elapsed + t_to_j + t_j_start) > max_duration + 1e-6
                mask_np[non_start & over_budget] = 0

                # Disponibilidad estocástica (vectorizado) — mismo esquema que RoutingEnv.
                # No se aplica cuando current_node == start_node (primer paso del beam).
                if beam["lane_exists"] is not None and current_node != start_node:
                    lane_absent = (beam["lane_exists"] == 0)
                    mask_np[non_start & lane_absent] = 0

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
                    step_time = float(time_matrix[current_node, next_node])
                    if time_elapsed + step_time > max_duration + 1e-6 and next_node != start_node:
                        continue

                    step_reward = float(rm_day[current_node, next_node])
                    new_time_elapsed = time_elapsed + step_time

                    # Sorteo de disponibilidad para next_node al llegar.
                    # Seed idéntico al de RoutingEnv: f(start_day_idx, node, arrival_day).
                    if avail_prob_arr is not None:
                        child_arrival_day  = min(
                            start_day_idx + int(new_time_elapsed // 14), max_day
                        )
                        child_lane_exists = draw_lane_availability(
                            start_day_idx, next_node, child_arrival_day,
                            avail_prob_arr, self.num_nodes,
                        )
                    else:
                        child_lane_exists = None

                    new_visited_set   = set(visited_set)
                    new_visited_inter = set(beam["visited_inter"])
                    new_visited_set.add(next_node)
                    if next_node != start_node:
                        new_visited_inter.add(next_node)

                    next_beams.append({
                        "current_node":  next_node,
                        "time_elapsed":  new_time_elapsed,
                        "visited_set":   new_visited_set,
                        "visited_inter": new_visited_inter,
                        "route":         beam["route"] + [next_node],
                        "total_reward":  beam["total_reward"] + step_reward,
                        "returned_home": next_node == start_node,
                        "lane_exists":   child_lane_exists,
                    })

            if not next_beams:
                break

            if critic is not None:
                def _score(b):
                    if b["returned_home"]:
                        v = 0.0
                    else:
                        b_day_offset = int(b["time_elapsed"] // 14)
                        b_day_idx    = min(start_day_idx + b_day_offset, max_day)
                        b_rm_day     = get_rm(b_day_idx)
                        v = self.estimate_value(
                            critic, b["current_node"], start_node, b["visited_set"],
                            b["time_elapsed"], step + 1,
                            b_rm_day, time_matrix, distance_arr,
                            max_duration, ltr_stack, trucks_stack, b_day_idx,
                            avail_prob_arr=avail_prob_arr,
                            reward_global_p95=reward_global_p95,
                        )
                    return b["total_reward"] + value_coef * v * REWARD_SCALE_FACTOR

                next_beams.sort(key=_score, reverse=True)
            else:
                next_beams.sort(key=lambda b: b["total_reward"], reverse=True)
            beams = next_beams[:beam_width]

            if all(b["returned_home"] for b in beams):
                break

        for beam in beams:
            if not beam["returned_home"] and beam["current_node"] != start_node:
                cn         = beam["current_node"]
                day_offset = int(beam["time_elapsed"] // 14)
                rm_day     = get_rm(min(start_day_idx + day_offset, max_day))
                t_ret = float(time_matrix[cn, start_node])
                r_ret = float(rm_day[cn, start_node])
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
    def generate_route_sampling(
        self,
        start_node:        int,
        start_day_idx:     int,
        time_matrix,
        rate_stack:        np.ndarray,
        loads_stack:       np.ndarray,
        distance_arr:      np.ndarray,
        diesel_arr:        np.ndarray,
        n_samples:         int,
        max_duration:      float      = MAX_DURATION,
        ltr_stack:         np.ndarray = None,   # [num_nodes, 120]
        trucks_stack:      np.ndarray = None,   # [num_nodes, 120, 3]
        avail_prob_arr:    np.ndarray = None,   # [num_nodes, num_nodes]
        reward_global_p95: float      = 1.0,
        temperature:       float      = 1.0,
    ):
        """Decodificación por muestreo: n_samples rollouts independientes.

        Misma semántica de día dinámico, máscara y retorno forzado que
        beam_search_dynamic (mismo manejo de over_budget, lane availability
        y forced return). La diferencia es la selección de acción: en vez
        de top-k determinista sobre los logits, cada muestra samplea de
        Categorical(softmax(logits / temperature)) sobre los nodos válidos.

        Los n_samples se vectorizan como batch B=n_samples en el encoder
        (un solo forward por paso para todas las muestras), por lo que el
        costo escala como pasos × forward(B=n_samples) en vez de
        n_samples × pasos × forward(B=1).

        Retorna la mejor ruta (por total_reward) entre las n_samples, con
        la misma definición de validez que beam_search_dynamic: returned_home,
        empieza y termina en start_node.

        Retorna
        -------
        route        : list[int] | None
        total_reward : float
        time_elapsed : float
        """
        from problem_data import build_rm_pen_stack

        self.eval()
        B      = n_samples
        N      = self.num_nodes
        device = self.device

        rm_pen_stack = build_rm_pen_stack(
            rate_stack, loads_stack, distance_arr, diesel_arr
        )   # (num_days, N, N) float32
        max_day = rm_pen_stack.shape[0] - 1

        time_matrix_arr = (
            time_matrix.to_numpy(dtype=float)
            if hasattr(time_matrix, "to_numpy")
            else np.asarray(time_matrix, dtype=float)
        )

        start_node_arr = np.full(B, start_node, dtype=np.int64)
        current_node   = np.full(B, start_node, dtype=np.int64)
        time_elapsed   = np.zeros(B, dtype=np.float32)
        step_count     = np.zeros(B, dtype=np.int64)
        visited_mask   = np.zeros((B, N), dtype=bool)
        visited_mask[:, start_node] = True
        routes         = [[start_node] for _ in range(B)]
        total_reward   = np.zeros(B, dtype=np.float64)
        returned_home  = np.zeros(B, dtype=bool)
        active         = np.ones(B, dtype=bool)

        if avail_prob_arr is not None:
            init_arrival_day = min(start_day_idx, max_day)
            init_row = draw_lane_availability(
                start_day_idx, start_node, init_arrival_day, avail_prob_arr, N,
            )
            lane_exists = np.tile(init_row, (B, 1))   # (B, N)
        else:
            lane_exists = None

        non_start = np.arange(N)[None, :] != start_node_arr[:, None]   # (B, N), constant

        for step in range(N):
            if not active.any():
                break

            day_idx = np.minimum(
                start_day_idx + (time_elapsed // 14).astype(np.int64), max_day
            )   # (B,)

            node_feats = build_node_features_batch(
                current_node, start_node_arr, visited_mask,
                rm_pen_stack, time_matrix_arr, distance_arr,
                day_idx, N, max_duration,
                trucks_stack_raw=trucks_stack,
                avail_prob_arr=avail_prob_arr,
                reward_global_p95=reward_global_p95,
            )   # (B, N, 8)
            temporal = build_temporal_features_batch(
                time_elapsed, step_count, max_duration, N
            )   # (B, 3)
            market = (
                build_market_features_batch(current_node, day_idx, ltr_stack)
                if ltr_stack is not None
                else np.zeros((B, 1), dtype=np.float32)
            )   # (B, 1)

            feats_t    = torch.from_numpy(node_feats).to(device)
            temporal_t = torch.from_numpy(temporal).to(device)
            market_t   = torch.from_numpy(market).to(device)

            embeddings, graph_emb = self.encoder(feats_t)             # (B,N,d_h), (B,d_h)
            check_tensor("generate_route_sampling embeddings", embeddings)
            current_node_t = torch.from_numpy(current_node).to(device)
            current_emb    = embeddings[torch.arange(B, device=device), current_node_t]
            h_t = self.context_net(graph_emb, current_emb, temporal_t, market_t)

            # ── Máscara: idéntica lógica a beam_search_dynamic, vectorizada ──
            mask_np = np.ones((B, N), dtype=np.int8)
            mask_np[np.arange(B), current_node] = 0                       # self-loop
            mask_np[visited_mask & non_start] = 0                         # visitados

            t_to_j      = time_matrix_arr[current_node, :]                # (B, N)
            t_j_start   = time_matrix_arr[:, start_node_arr].T            # (B, N)
            over_budget = (
                time_elapsed[:, None] + t_to_j + t_j_start
            ) > max_duration + 1e-6
            mask_np[non_start & over_budget] = 0

            if lane_exists is not None:
                not_at_start = (current_node != start_node_arr)[:, None]
                lane_absent  = (lane_exists == 0)
                mask_np[not_at_start & lane_absent & non_start] = 0

            valid_non_start_count = (mask_np * non_start).sum(axis=1)
            not_at_start_flag     = current_node != start_node_arr
            force_return = (valid_non_start_count == 0) & not_at_start_flag
            stuck        = (mask_np.sum(axis=1) == 0)   # no valid action at all (dead end at depot)

            # Garantizar al menos una posición válida por fila SOLO para evitar
            # NaN en glimpse/softmax sobre filas inactivas/atascadas/forzadas;
            # la decisión real para esas filas no depende de este forward.
            mask_for_logits = mask_np.copy()
            guard_rows = force_return | stuck | (~active)
            mask_for_logits[guard_rows, start_node_arr[guard_rows]] = 1

            mask_t    = torch.from_numpy(mask_for_logits).to(device)
            bool_mask = (mask_t == 0)
            logits    = self.decoder._logits(h_t, embeddings, bool_mask)   # (B, N)
            logits    = logits / temperature
            probs     = torch.softmax(logits, dim=-1)
            probs     = torch.clamp(probs, min=1e-8)
            dist      = torch.distributions.Categorical(probs=probs)
            sampled   = dist.sample().cpu().numpy()                       # (B,)

            next_node = np.where(force_return, start_node_arr, sampled)

            step_time   = time_matrix_arr[current_node, next_node]
            step_reward = rm_pen_stack[day_idx, current_node, next_node].astype(np.float64)
            new_time_elapsed = time_elapsed + step_time
            just_returned    = (next_node == start_node_arr) & active

            update = active & ~stuck
            for b in np.where(update)[0]:
                routes[b].append(int(next_node[b]))
            visited_mask[np.arange(B), next_node] = np.where(
                update, True, visited_mask[np.arange(B), next_node]
            )
            current_node = np.where(update, next_node, current_node)
            time_elapsed = np.where(update, new_time_elapsed, time_elapsed)
            step_count   = np.where(update, step_count + 1, step_count)
            total_reward = total_reward + np.where(update, step_reward, 0.0)
            returned_home = returned_home | (just_returned & update)

            if lane_exists is not None:
                child_arrival_day = np.minimum(
                    start_day_idx + (new_time_elapsed // 14).astype(np.int64), max_day
                )
                new_lane_rows = np.stack([
                    draw_lane_availability(
                        start_day_idx, int(next_node[b]), int(child_arrival_day[b]),
                        avail_prob_arr, N,
                    )
                    for b in range(B)
                ])
                lane_exists = np.where(update[:, None], new_lane_rows, lane_exists)

            active = active & update & ~returned_home

        # ── Retorno forzado para muestras que no cerraron el ciclo ──────────
        unfinished = np.where(~returned_home)[0]
        if len(unfinished) > 0:
            day_idx_final = np.minimum(
                start_day_idx + (time_elapsed[unfinished] // 14).astype(np.int64), max_day
            )
            t_ret = time_matrix_arr[current_node[unfinished], start_node]
            r_ret = rm_pen_stack[day_idx_final, current_node[unfinished], start_node]
            feasible = time_elapsed[unfinished] + t_ret <= max_duration + 1e-6
            for k, b in enumerate(unfinished):
                if feasible[k] and current_node[b] != start_node:
                    time_elapsed[b] += t_ret[k]
                    total_reward[b] += r_ret[k]
                    routes[b].append(start_node)
                    returned_home[b] = True

        best_idx    = None
        best_reward = -np.inf
        for b in range(B):
            route = routes[b]
            is_valid = (
                returned_home[b]
                and len(route) > 1
                and route[0] == start_node
                and route[-1] == start_node
            )
            if is_valid and total_reward[b] > best_reward:
                best_reward = total_reward[b]
                best_idx    = b

        if best_idx is None:
            return None, -np.inf, np.inf

        return routes[best_idx], float(total_reward[best_idx]), float(time_elapsed[best_idx])

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
        action_mask:            np.ndarray,        # (N,) int8 de RoutingEnv
        ltr_stack:              np.ndarray = None,
        trucks_stack:           np.ndarray = None,
        day_idx:                int        = 0,
        avail_prob_arr:         np.ndarray = None,
        reward_global_p95:      float      = 1.0,
    ) -> int:
        """Selección determinista (argmax de logits) para evaluación env-based."""
        embeddings, h_t, _, _, _ = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration, ltr_stack, trucks_stack, day_idx,
            avail_prob_arr=avail_prob_arr,
            reward_global_p95=reward_global_p95,
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
        action_mask:            np.ndarray,        # (N,) int8 de RoutingEnv
        ltr_stack:              np.ndarray = None,
        trucks_stack:           np.ndarray = None,
        day_idx:                int        = 0,
        avail_prob_arr:         np.ndarray = None,
        reward_global_p95:      float      = 1.0,
    ):
        """
        Paso estocástico para entrenamiento PPO.

        Retorna
        -------
        action   : int
        log_prob : Tensor escalar
        entropy  : Tensor escalar
        """
        embeddings, h_t, _, _, _ = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration, ltr_stack, trucks_stack, day_idx,
            avail_prob_arr=avail_prob_arr,
            reward_global_p95=reward_global_p95,
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
        action_mask:            np.ndarray,        # (N,) int8 de RoutingEnv
        ltr_stack:              np.ndarray = None,
        trucks_stack:           np.ndarray = None,
        day_idx:                int        = 0,
        avail_prob_arr:         np.ndarray = None,
        reward_global_p95:      float      = 1.0,
    ):
        """
        Paso estocástico para recolección PPO: samplea acción y estima valor.

        Retorna
        -------
        action     : int
        log_prob   : Tensor escalar
        entropy    : Tensor escalar
        value      : float
        node_feats : np.ndarray (N, N_NODE_FEATURES)
        temporal   : np.ndarray (3,)
        market     : np.ndarray (1,)
        """
        embeddings, h_t, node_feats, temporal, market = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration, ltr_stack, trucks_stack, day_idx,
            avail_prob_arr=avail_prob_arr,
            reward_global_p95=reward_global_p95,
        )
        critic_val = critic(h_t)                                            # (1,1) on device
        mask_t     = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
        action_t, log_prob, entropy = self.decoder.act(h_t, embeddings, mask_t)
        # action is int64 — extract with .item() to avoid float32 cast and
        # the NumPy ≥1.25 deprecation of int(ndim>0 array).
        # log_prob and critic_val are both float32 scalars: batch them into one
        # CPU transfer (2 syncs total instead of the original 3).
        action = int(action_t.item())
        sync   = torch.stack([log_prob.view(1), critic_val.view(1)]).cpu().numpy()
        return action, float(sync[0]), entropy.squeeze(0), float(sync[1]), node_feats, temporal, market

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
        ltr_stack:              np.ndarray = None,
        trucks_stack:           np.ndarray = None,
        day_idx:                int        = 0,
        avail_prob_arr:         np.ndarray = None,
        reward_global_p95:      float      = 1.0,
    ) -> float:
        """Estima V(s) para bootstrap en episodios truncados."""
        _, h_t, _, _, _ = self._encode_step(
            current_node, start_node, visited_set,
            time_elapsed, step_count,
            reward_matrix_penalized, time_matrix, distance_arr,
            max_duration, ltr_stack, trucks_stack, day_idx,
            avail_prob_arr=avail_prob_arr,
            reward_global_p95=reward_global_p95,
        )
        return critic(h_t).item()
