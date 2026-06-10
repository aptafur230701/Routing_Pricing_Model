"""
routing_env.py
==============
RoutingEnv: entorno Gymnasium para el problema de ruteo estocástico con
revelación post-decisión de disponibilidad de lanes (Stochastic DVRP).

MDP estocástico — revelación post-decisión
------------------------------------------
Cuando el camión decide moverse al nodo X, NO conoce qué lanes saldrán de X
en la fecha de llegada; esa disponibilidad se revela AL LLEGAR.
Cada lane saliente de X se modela como un Bernoulli(avail_prob_arr[X, j]).
El sorteo es determinista dado (start_day_idx, node, arrival_day), donde
arrival_day = start_day_idx + int(time_elapsed // 14), clampado al último día
disponible. Esto garantiza que dos rutas que llegan al mismo nodo en la misma
fecha de calendario siempre ven la misma realización del mundo.

API Gymnasium moderna (estricta):
  obs, info                                = env.reset(seed=..., options={"start_node": k})
  obs, reward, terminated, truncated, info = env.step(action)

Condiciones de terminación (exclusivamente):
  A) El camión regresa al start_node       → terminated=True
  B) No existe ninguna acción factible     → terminated=True (red de seguridad)
  truncated siempre es False — ya no hay límite de pasos.

La máscara de acciones (info["action_mask"]) excluye:
  · self-loops
  · nodos intermedios ya visitados
  · nodos desde los que no es posible regresar al depot dentro de max_duration
  · lanes salientes que no existen según el sorteo de disponibilidad del día
  start_node siempre permanece válido (la violación de tiempo al regresar la
  detecta _check_termination, no la máscara).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    MAX_DURATION,
    REWARD_SCALE_FACTOR,
    RETURN_SUCCESS_BONUS,
    TIME_VIOLATION_PENALTY,
)
from state_features import get_state_size, build_state
from problem_data import build_day_matrices, draw_lane_availability


class RoutingEnv(gym.Env):
    """
    Entorno Gymnasium para el ruteo estocástico de camiones (SDVRP).

    El agente selecciona el nodo destino en cada paso. El episodio termina
    cuando el camión cierra el ciclo regresando al nodo de inicio (condición A)
    o cuando no existe ninguna acción factible (condición B — red de seguridad).
    Ya no existe truncación por número de pasos.

    Parámetros
    ----------
    time_matrix    : pd.DataFrame | np.ndarray  (num_nodes × num_nodes)
    rate_stack     : np.ndarray  [num_days, num_nodes, num_nodes]
    loads_stack    : np.ndarray  [num_days, num_nodes, num_nodes]
    distance_arr   : np.ndarray  (num_nodes × num_nodes)
    diesel_arr     : np.ndarray  (num_nodes × num_nodes)
    avail_prob_arr : np.ndarray  (num_nodes × num_nodes) float32
                     Prior Bernoulli de existencia por arco; producido por
                     load_matrices() / problem_data.draw_lane_availability().
    num_nodes      : int
    max_duration   : float, opcional (default: MAX_DURATION)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        time_matrix,
        rate_stack:     np.ndarray,
        loads_stack:    np.ndarray,
        distance_arr:   np.ndarray,
        diesel_arr:     np.ndarray,
        avail_prob_arr: np.ndarray,
        num_nodes:      int,
        max_duration:   float = MAX_DURATION,
    ):
        super().__init__()

        self._time_matrix   = time_matrix
        self._rate_stack    = rate_stack
        self._loads_stack   = loads_stack
        self._distance_arr  = distance_arr
        self._diesel_arr    = diesel_arr
        self._avail_prob_arr = avail_prob_arr
        self.num_nodes      = num_nodes
        self.max_duration   = max_duration

        state_size = get_state_size(num_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32,
        )

        self._start_node:         int      = None
        self._current_node:       int      = None
        self._time_elapsed:       float    = None
        self._visited_set:        set      = None
        self._step_count:         int      = None
        self._start_day_idx:      int      = None
        self._current_lane_exists: np.ndarray = None

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida del entorno
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)

        if options is not None and "start_node" in options:
            self._start_node = int(options["start_node"])
        else:
            self._start_node = int(self.np_random.integers(0, self.num_nodes))

        if options is not None and "start_day_idx" in options:
            self._start_day_idx = int(options["start_day_idx"])
        else:
            self._start_day_idx = 0

        self._current_node = self._start_node
        self._time_elapsed = 0.0
        self._visited_set  = {self._start_node}
        self._step_count   = 0

        # Sorteo inicial de disponibilidad para las lanes salientes del nodo de inicio.
        # time_elapsed=0 → arrival_day = start_day_idx (sin offset).
        max_day = self._rate_stack.shape[0] - 1
        arrival_day = min(self._start_day_idx, max_day)
        self._current_lane_exists = draw_lane_availability(
            self._start_day_idx, self._start_node, arrival_day,
            self._avail_prob_arr, self.num_nodes,
        )

        obs  = self._build_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        assert self._current_node is not None, (
            "El entorno no está inicializado. Llama env.reset() antes de env.step()."
        )

        next_node = int(action)

        # 1. Dinámica de transición
        next_time = self._transition(next_node)

        # 2. Condición A: retorno al depot o violación de tiempo
        terminated = self._check_termination(next_node, next_time)

        # 3. Recompensa del paso
        reward = self._compute_reward(next_node, next_time, terminated)

        # 4. Actualizar estado interno
        self._visited_set  = self._visited_set | {next_node}
        self._current_node = next_node
        self._time_elapsed = next_time
        self._step_count  += 1

        # 5. Revelar disponibilidad de lanes desde next_node (post-decisión).
        # Seed basado en (start_day_idx, node, arrival_day): dos rutas que llegan
        # al mismo nodo el mismo día de calendario ven la misma realización.
        max_day     = self._rate_stack.shape[0] - 1
        arrival_day = min(self._start_day_idx + int(next_time // 14), max_day)
        self._current_lane_exists = draw_lane_availability(
            self._start_day_idx, next_node, arrival_day,
            self._avail_prob_arr, self.num_nodes,
        )

        # 6. Observación y máscara del siguiente estado
        next_obs = self._build_obs()
        next_mask = self._get_action_mask()

        # Condición B: si la máscara quedó vacía, terminar limpiamente
        if not terminated and next_mask.sum() == 0:
            terminated = True

        info = {"action_mask": next_mask}
        return next_obs, float(reward), terminated, False, info

    # ─────────────────────────────────────────────────────────────────────────
    # Componentes internos
    # ─────────────────────────────────────────────────────────────────────────

    def _transition(self, next_node: int) -> float:
        step_time = (
            self._time_matrix.iloc[self._current_node, next_node]
            if hasattr(self._time_matrix, "iloc")
            else float(self._time_matrix[self._current_node][next_node])
        )
        return self._time_elapsed + float(step_time)

    def _get_current_day_idx(self) -> int:
        day_offset = int(self._time_elapsed // 14)
        raw = self._start_day_idx + day_offset
        return min(raw, self._rate_stack.shape[0] - 1)

    def _compute_reward(
        self, next_node: int, next_time: float, terminated: bool
    ) -> float:
        current_day_idx = self._get_current_day_idx()
        _, reward_matrix_penalized_step = build_day_matrices(
            self._rate_stack[current_day_idx],
            self._loads_stack[current_day_idx],
            self._distance_arr,
            self._diesel_arr,
        )
        raw_reward = float(reward_matrix_penalized_step.iloc[self._current_node, next_node])
        step_reward = float(raw_reward) / REWARD_SCALE_FACTOR

        terminal_reward = 0.0
        if terminated:
            if next_node == self._start_node:
                if next_time <= self.max_duration:
                    n_intermediate = len(self._visited_set) - 1
                    time_util = next_time / self.max_duration
                    terminal_reward = (
                        RETURN_SUCCESS_BONUS
                        + (RETURN_SUCCESS_BONUS * 0.5) * (n_intermediate / (self.num_nodes - 1))
                        + (RETURN_SUCCESS_BONUS * 0.3) * time_util
                    )
                else:
                    terminal_reward = TIME_VIOLATION_PENALTY
            else:
                terminal_reward = TIME_VIOLATION_PENALTY

        # Penalización de callejón temporal — solo en pasos no terminales hacia intermedios
        temporal_warning = 0.0
        if not terminated and next_node != self._start_node:
            t_return = (
                float(self._time_matrix.iloc[next_node, self._start_node])
                if hasattr(self._time_matrix, "iloc")
                else float(self._time_matrix[next_node][self._start_node])
            )
            slack = self.max_duration - (next_time + t_return)
            if slack < 0:
                temporal_warning = 0.05 * TIME_VIOLATION_PENALTY

        return step_reward + terminal_reward + temporal_warning

    def _check_termination(self, next_node: int, next_time: float) -> bool:
        """
        Condición A: el camión cerró el ciclo o excedió el tiempo.
        Retorna solo terminated (bool). truncated siempre es False.
        """
        return (
            next_node == self._start_node
            or next_time > self.max_duration
        )

    def _get_action_mask(self) -> np.ndarray:
        """
        Máscara binaria de acciones válidas para el estado actual.

        Reglas (en orden de aplicación):
        · 0 para self-loop (nodo actual).
        · 0 para nodos intermedios ya visitados.
        · 0 para nodos intermedios desde los que no se puede regresar al depot
          dentro de max_duration (lookahead temporal).
        · 0 para lanes salientes que no existen según el sorteo estocástico de
          disponibilidad revelado al llegar al nodo actual.
        · start_node permanece siempre disponible (mask=1) para permitir retorno;
          si el tiempo al regresar excede max_duration, _check_termination lo detecta.
        """
        mask = np.ones(self.num_nodes, dtype=np.int8)

        # Self-loop
        mask[self._current_node] = 0

        # Intermedios visitados
        for v in self._visited_set:
            if v != self._start_node:
                mask[v] = 0

        # Lookahead temporal: excluir intermedios que agoten el presupuesto
        has_iloc = hasattr(self._time_matrix, "iloc")
        for j in range(self.num_nodes):
            if mask[j] == 1 and j != self._start_node:
                t_to_j = (
                    float(self._time_matrix.iloc[self._current_node, j])
                    if has_iloc
                    else float(self._time_matrix[self._current_node][j])
                )
                t_j_to_start = (
                    float(self._time_matrix.iloc[j, self._start_node])
                    if has_iloc
                    else float(self._time_matrix[j][self._start_node])
                )
                if self._time_elapsed + t_to_j + t_j_to_start > self.max_duration:
                    mask[j] = 0

        # Disponibilidad estocástica: excluir lanes que no existen en este sorteo.
        # start_node nunca se enmascara por disponibilidad — retorno siempre válido.
        # Cuando current_node == start_node (primer paso) no se aplica: el camión
        # ya está ahí, no "llega" — la incertidumbre solo aplica al transitar a nodos nuevos.
        if self._current_lane_exists is not None and self._current_node != self._start_node:
            for j in range(self.num_nodes):
                if mask[j] == 1 and j != self._start_node:
                    if self._current_lane_exists[j] == 0:
                        mask[j] = 0

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # API pública auxiliar
    # ─────────────────────────────────────────────────────────────────────────

    def get_valid_actions(self) -> list:
        """Índices de nodos con acción válida en el estado actual."""
        return [i for i, m in enumerate(self._get_action_mask()) if m == 1]

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de solo lectura
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def start_node(self) -> int:
        return self._start_node

    @property
    def current_node(self) -> int:
        return self._current_node

    @property
    def time_elapsed(self) -> float:
        return self._time_elapsed

    @property
    def visited_set(self) -> frozenset:
        return frozenset(self._visited_set)

    @property
    def current_day_idx(self) -> int:
        return self._get_current_day_idx()

    # ─────────────────────────────────────────────────────────────────────────
    # Auxiliar interno
    # ─────────────────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        return build_state(
            self._current_node,
            self._time_elapsed,
            self._visited_set,
            self._step_count,
            self.max_duration,
            self.num_nodes,
        )
