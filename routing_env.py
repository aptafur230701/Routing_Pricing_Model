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
        # Cache: day_idx → reward_matrix_penalized (np.ndarray).
        # Persists across episodes since build_day_matrices is deterministic.
        self._rm_pen_cache:       dict     = {}

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
        return self._time_elapsed + float(self._time_matrix[self._current_node, next_node])

    def _get_current_day_idx(self) -> int:
        day_offset = int(self._time_elapsed // 14)
        raw = self._start_day_idx + day_offset
        return min(raw, self._rate_stack.shape[0] - 1)

    def _get_rm_pen(self, day_idx: int) -> np.ndarray:
        if day_idx not in self._rm_pen_cache:
            _, self._rm_pen_cache[day_idx] = build_day_matrices(
                self._rate_stack[day_idx],
                self._loads_stack[day_idx],
                self._distance_arr,
                self._diesel_arr,
            )
        return self._rm_pen_cache[day_idx]

    def _compute_reward(
        self, next_node: int, next_time: float, terminated: bool
    ) -> float:
        current_day_idx = self._get_current_day_idx()
        rm_pen = self._get_rm_pen(current_day_idx)
        raw_reward = float(rm_pen[self._current_node, next_node])
        step_reward = raw_reward / REWARD_SCALE_FACTOR

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
            t_return = float(self._time_matrix[next_node, self._start_node])
            if self.max_duration < next_time + t_return:
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

        # Intermedios visitados (vectorizado — start_node siempre se omite)
        visited_inter = [v for v in self._visited_set if v != self._start_node]
        if visited_inter:
            mask[visited_inter] = 0

        # Lookahead temporal (vectorizado): excluir intermedios que agoten el presupuesto.
        # start_node queda exento — su retorno es siempre válido en la máscara.
        non_start   = np.arange(self.num_nodes) != self._start_node
        t_to_j      = self._time_matrix[self._current_node]      # (N,) current→j
        t_j_to_dep  = self._time_matrix[:, self._start_node]     # (N,) j→depot
        over_budget = (self._time_elapsed + t_to_j + t_j_to_dep) > self.max_duration
        mask[non_start & over_budget] = 0

        # Disponibilidad estocástica (vectorizado): excluir lanes que no existen.
        # start_node nunca se enmascara — retorno siempre válido.
        # Cuando current_node == start_node (primer paso) no se aplica.
        if self._current_lane_exists is not None and self._current_node != self._start_node:
            lane_absent = (self._current_lane_exists == 0)
            mask[non_start & lane_absent] = 0

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


class VectorRoutingEnv:
    """
    Vectorized version of RoutingEnv: runs B episodes in parallel as a single batch.

    State is stored as numpy arrays of shape (B, ...) instead of scalars/sets.
    Used exclusively by the PPO collection phase in am_training.py.
    Does NOT implement the Gymnasium API — it has a simpler custom interface.

    Parámetros
    ----------
    rm_pen_stack : np.ndarray (num_train_days, N, N) float32
                   Penalized reward matrices precomputed by build_rm_pen_stack().
                   Passed in from training to avoid recomputing per step.
    """

    def __init__(
        self,
        time_matrix,
        rate_stack:      np.ndarray,
        loads_stack:     np.ndarray,
        distance_arr:    np.ndarray,
        diesel_arr:      np.ndarray,
        avail_prob_arr:  np.ndarray,
        rm_pen_stack:    np.ndarray,
        num_nodes:       int,
        max_duration:    float = MAX_DURATION,
    ):
        self._time_matrix_arr = (
            time_matrix.to_numpy(dtype=float)
            if hasattr(time_matrix, "to_numpy")
            else np.asarray(time_matrix, dtype=float)
        )
        self._rate_stack     = rate_stack
        self._avail_prob_arr = avail_prob_arr
        self._rm_pen_stack   = rm_pen_stack   # (num_train_days, N, N)
        self.num_nodes       = num_nodes
        self.max_duration    = max_duration

        # Initialized by reset()
        self._B:                   int        = 0
        self._start_node:          np.ndarray = None   # (B,) int64
        self._current_node:        np.ndarray = None   # (B,) int64
        self._time_elapsed:        np.ndarray = None   # (B,) float32
        self._visited_mask:        np.ndarray = None   # (B, N) bool
        self._step_count:          np.ndarray = None   # (B,) int64
        self._start_day_idx:       np.ndarray = None   # (B,) int64
        self._current_lane_exists: np.ndarray = None   # (B, N) int8
        self._active:              np.ndarray = None   # (B,) bool

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        start_nodes:    np.ndarray,   # (B,) int64
        start_day_idxs: np.ndarray,   # (B,) int64
    ) -> np.ndarray:                  # masks (B, N) int8
        B = len(start_nodes)
        N = self.num_nodes
        self._B = B

        self._start_node    = np.asarray(start_nodes,    dtype=np.int64)
        self._current_node  = self._start_node.copy()
        self._time_elapsed  = np.zeros(B, dtype=np.float32)
        self._visited_mask  = np.zeros((B, N), dtype=bool)
        self._visited_mask[np.arange(B), self._start_node] = True
        self._step_count    = np.zeros(B, dtype=np.int64)
        self._start_day_idx = np.asarray(start_day_idxs, dtype=np.int64)
        self._active        = np.ones(B, dtype=bool)

        # Initial lane availability: one draw_lane_availability call per row
        max_day = self._rate_stack.shape[0] - 1
        lane_rows = []
        for b in range(B):
            arrival_day = min(int(self._start_day_idx[b]), max_day)
            lane_rows.append(draw_lane_availability(
                int(self._start_day_idx[b]),
                int(self._start_node[b]),
                arrival_day,
                self._avail_prob_arr,
                N,
            ))
        self._current_lane_exists = np.stack(lane_rows)   # (B, N) int8

        return self._compute_masks()

    def step(
        self,
        actions: np.ndarray,   # (B,) int64
    ) -> tuple:                # (rewards (B,), terminated (B,), next_masks (B, N))
        """
        Execute one step for all B episodes.

        Only active rows (self._active) are processed; inactive rows receive
        reward=0 and terminated=False (they are already done).

        Returns terminated = True only for rows that terminate in THIS step
        (condition A: depot return or time violation; condition B: empty mask).
        Truncation (step ceiling exceeded) is detected outside, in the training loop.
        """
        B   = self._B
        N   = self.num_nodes
        actions = np.asarray(actions, dtype=np.int64)

        # ── Time transition ───────────────────────────────────────────────────
        t_step   = self._time_matrix_arr[self._current_node, actions].astype(np.float32)
        next_time = self._time_elapsed + t_step

        # ── Condition A termination (only for active rows) ────────────────────
        cond_a      = (actions == self._start_node) | (next_time > self.max_duration)
        terminated_a = self._active & cond_a

        # ── Reward (computed BEFORE state update, using current node/day) ──────
        max_rm_day = self._rm_pen_stack.shape[0] - 1
        day_idx = np.minimum(
            self._start_day_idx + (self._time_elapsed // 14).astype(np.int64),
            max_rm_day,
        )   # (B,)

        step_reward = (
            self._rm_pen_stack[day_idx, self._current_node, actions].astype(np.float32)
            / REWARD_SCALE_FACTOR
        )

        # Terminal reward components (using visited_mask BEFORE the state update)
        n_intermediate = self._visited_mask.sum(axis=1).astype(np.float32) - 1   # (B,)
        time_util      = next_time / self.max_duration

        success   = self._active & (actions == self._start_node) & (next_time <= self.max_duration)
        time_fail = self._active & cond_a & ~success

        terminal_reward = np.zeros(B, dtype=np.float32)
        if success.any():
            denom = max(N - 1, 1)
            terminal_reward[success] = (
                RETURN_SUCCESS_BONUS
                + (RETURN_SUCCESS_BONUS * 0.5) * (n_intermediate[success] / denom)
                + (RETURN_SUCCESS_BONUS * 0.3) * time_util[success]
            )
        if time_fail.any():
            terminal_reward[time_fail] = TIME_VIOLATION_PENALTY

        # Temporal dead-end warning (non-terminal, non-depot active steps)
        t_return = self._time_matrix_arr[actions, self._start_node].astype(np.float32)   # (B,)
        warn_mask = (
            self._active & ~cond_a
            & (actions != self._start_node)
            & (self.max_duration < next_time + t_return)
        )
        temporal_warning = np.where(warn_mask, 0.05 * TIME_VIOLATION_PENALTY, 0.0).astype(np.float32)

        rewards = np.where(
            self._active,
            step_reward + terminal_reward + temporal_warning,
            0.0,
        ).astype(np.float32)

        # ── State update (all active rows, including those terminating by cond A) ─
        self._visited_mask[np.arange(B), actions] |= self._active
        self._current_node = np.where(self._active, actions, self._current_node).astype(np.int64)
        self._time_elapsed  = np.where(self._active, next_time, self._time_elapsed)
        self._step_count    = np.where(self._active, self._step_count + 1, self._step_count)

        # ── Lane revelation (active rows NOT terminated by condition A) ───────
        max_day = self._rate_stack.shape[0] - 1
        for b in np.where(self._active & ~cond_a)[0]:
            arrival_day = min(
                int(self._start_day_idx[b]) + int(self._time_elapsed[b] // 14),
                max_day,
            )
            self._current_lane_exists[b] = draw_lane_availability(
                int(self._start_day_idx[b]),
                int(self._current_node[b]),
                arrival_day,
                self._avail_prob_arr,
                N,
            )

        # ── Update active after condition A ───────────────────────────────────
        self._active = self._active & ~cond_a

        # ── Compute next masks ────────────────────────────────────────────────
        next_masks = self._compute_masks()

        # ── Condition B: zero valid actions for still-active rows ─────────────
        cond_b = (next_masks.sum(axis=1) == 0) & self._active
        self._active = self._active & ~cond_b

        terminated = terminated_a | cond_b

        # Ensure inactive rows have at least one valid action (prevents _forward crash)
        all_zero = next_masks.sum(axis=1) == 0
        next_masks[all_zero, 0] = 1

        return rewards, terminated, next_masks

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_masks(self) -> np.ndarray:
        """Compute action masks for all B episodes. Returns (B, N) int8."""
        B = self._B
        N = self.num_nodes
        mask = np.ones((B, N), dtype=np.int8)

        # non-depot indicator per (b, j) pair
        non_depot = np.arange(N)[None, :] != self._start_node[:, None]   # (B, N)

        # Visited intermediates
        mask[self._visited_mask & non_depot] = 0
        # Depot always valid (will be re-confirmed after each rule)
        mask[np.arange(B), self._start_node] = 1

        # Temporal lookahead
        t_to_j     = self._time_matrix_arr[self._current_node, :]         # (B, N)
        t_j_to_dep = self._time_matrix_arr[:, self._start_node].T         # (B, N)
        over_budget = (
            self._time_elapsed[:, None] + t_to_j + t_j_to_dep
        ) > self.max_duration                                              # (B, N)
        mask[over_budget & non_depot] = 0
        mask[np.arange(B), self._start_node] = 1

        # Stochastic lane availability (skip when at depot/start_node)
        not_at_start = (self._current_node != self._start_node)[:, None]   # (B, 1)
        lane_absent  = (self._current_lane_exists == 0)                     # (B, N)
        mask[not_at_start & lane_absent & non_depot] = 0
        mask[np.arange(B), self._start_node] = 1

        # Self-loop: applied LAST so it takes precedence over all depot restorations.
        # When current_node == start_node (first step of episode), this masks out
        # the depot itself, matching the scalar RoutingEnv._get_action_mask() behavior.
        mask[np.arange(B), self._current_node] = 0

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # Read-only properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def start_node(self) -> np.ndarray:
        return self._start_node

    @property
    def current_node(self) -> np.ndarray:
        return self._current_node

    @property
    def time_elapsed(self) -> np.ndarray:
        return self._time_elapsed

    @property
    def step_count(self) -> np.ndarray:
        return self._step_count

    @property
    def visited_mask(self) -> np.ndarray:
        return self._visited_mask

    @property
    def current_day_idx(self) -> np.ndarray:
        max_day = self._rm_pen_stack.shape[0] - 1
        return np.minimum(
            self._start_day_idx + (self._time_elapsed // 14).astype(np.int64),
            max_day,
        )
