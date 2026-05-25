"""
routing_env.py
==============
RoutingEnv: entorno Gymnasium formal para el problema de ruteo estocástico (SDVRP).

Encapsula la dinámica de transición, el cálculo de recompensas y las condiciones
de terminación que anteriormente estaban dispersas en training.py y tuning.py.

API Gymnasium moderna (estricta):
  obs, info                           = env.reset(seed=..., options={"start_node": k})
  obs, reward, terminated, truncated, info = env.step(action)

El diccionario ``info`` siempre incluye la clave ``action_mask`` (np.int8, 1 = válido,
0 = prohibido) para que DQNAgent_Optimized.act() pueda aplicar enmascaramiento.

Separación arquitectónica interna (sin lógica monolítica en step):
  _transition()         — dinámica de movimiento y tiempo
  _compute_reward()     — recompensa del arco (estocástica o determinista) + bonos terminales
  _check_termination()  — condiciones de término (terminated / truncated)
  _get_action_mask()    — máscara binaria de acciones válidas

Compatibilidad:
  · No modifica agent.py, networks.py, replay_buffer.py, config.py,
    state.py, environment.py, evaluation.py ni Solvers.py.
  · Las funciones build_state() y sample_stochastic_reward() se reutilizan
    directamente desde sus módulos originales.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    STOCHASTIC_MODE,
    MAX_STEPS_PER_EPISODE,
    MAX_DURATION,
    REWARD_SCALE_FACTOR,
    RETURN_SUCCESS_BONUS,
    TIME_VIOLATION_PENALTY,
)
from state import get_state_size, build_state
from problem_data import sample_stochastic_reward


class RoutingEnv(gym.Env):
    """
    Entorno Gymnasium para el ruteo estocástico de camiones (SDVRP).

    El agente selecciona el nodo destino en cada paso. El episodio termina
    cuando el camión cierra el ciclo regresando al nodo de inicio (terminated)
    o cuando se alcanza el límite de pasos (truncated).

    Parámetros
    ----------
    time_matrix              : pd.DataFrame | np.ndarray  (num_nodes × num_nodes)
        Tiempos de viaje entre pares de nodos (horas).
    reward_matrix_penalized  : pd.DataFrame | np.ndarray  (diagonal = BIG_M_PENALTY)
        Recompensas brutas por arco; diagonal penalizada para evitar self-loops.
    noise_sigma              : float
        Desviación estándar del ruido estocástico en las recompensas.
        En modo determinista (STOCHASTIC_MODE=False) se ignora.
    num_nodes                : int
        Número de nodos del grafo de ruteo.
    max_steps                : int, opcional
        Límite de pasos por episodio. Por defecto: MAX_STEPS_PER_EPISODE.
    max_duration             : float, opcional
        Límite de tiempo de operación por episodio (horas). Por defecto: MAX_DURATION.

    Spaces
    ------
    action_space      : Discrete(num_nodes)
    observation_space : Box(float32, shape=(2 + num_nodes + 2,))
        Layout del vector de estado — ver state.py para el detalle completo.

    Info
    ----
    Cada llamada a reset() y step() retorna un dict ``info`` con:
        "action_mask" : np.ndarray[int8, shape=(num_nodes,)]
            1 = nodo permitido, 0 = nodo prohibido.

    Notas de escalabilidad
    ----------------------
    El entorno es completamente paramétrico: al construirlo con num_nodes=100
    y las matrices correspondientes escala automáticamente sin cambios de código.
    Los métodos privados son puntos de extensión naturales para agregar ventanas
    de tiempo dinámicas, precios de reserva (Pricing Network) o atención.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        time_matrix,
        reward_matrix_penalized,
        noise_sigma: float,
        num_nodes: int,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        max_duration: float = MAX_DURATION,
    ):
        super().__init__()

        # ── Datos del problema ────────────────────────────────────
        self._time_matrix = time_matrix
        self._reward_matrix_penalized = reward_matrix_penalized
        self._noise_sigma = noise_sigma
        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.max_duration = max_duration

        # ── Espacios de Gymnasium ─────────────────────────────────
        state_size = get_state_size(num_nodes)
        self.action_space = spaces.Discrete(num_nodes)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_size,),
            dtype=np.float32,
        )

        # ── Estado interno del episodio (inicializado en reset()) ─
        self._start_node: int = None
        self._current_node: int = None
        self._time_elapsed: float = None
        self._visited_set: set = None
        self._step_count: int = None

    # ─────────────────────────────────────────────────────────────────────────
    # Ciclo de vida del entorno 
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self, seed: int = None, options: dict = None):
        """
        Inicializa un nuevo episodio.

        Parámetros
        ----------
        seed    : int, opcional — semilla para reproducibilidad.
        options : dict, opcional
            Puede contener ``"start_node"`` (int) para fijar el nodo de inicio.
            Si se omite, el nodo se selecciona uniformemente al azar.

        Retorna
        -------
        obs  : np.ndarray[float32]  — vector de estado inicial.
        info : dict con ``"action_mask"`` (np.int8 array, shape=(num_nodes,)).
        """
        super().reset(seed=seed)

        if options is not None and "start_node" in options:
            self._start_node = int(options["start_node"])
        else:
            self._start_node = int(self.np_random.integers(0, self.num_nodes))

        self._current_node = self._start_node
        self._time_elapsed = 0.0
        self._visited_set = {self._start_node}
        self._step_count = 0

        obs = self._build_obs()
        info = {"action_mask": self._get_action_mask()}
        return obs, info

    def step(self, action: int):
        """
        Ejecuta la acción (índice del nodo destino) en el entorno.

        El flujo interno sigue la separación arquitectónica:
          1. _transition()        — calcula el nuevo tiempo
          2. _check_termination() — evalúa terminated / truncated
          3. _compute_reward()    — calcula la recompensa del paso
          4. Actualización del estado interno
          5. Construcción de la observación y máscara siguientes

        Parámetros
        ----------
        action : int — índice del nodo destino (0..num_nodes-1).

        Retorna
        -------
        next_obs   : np.ndarray[float32]
        reward     : float
        terminated : bool — True si el camión cerró el ciclo o violó tiempo.
        truncated  : bool — True si se alcanzó el límite máximo de pasos.
        info       : dict con ``"action_mask"`` para el siguiente estado.
        """
        assert self._current_node is not None, (
            "El entorno no está inicializado. Llama env.reset() antes de env.step()."
        )

        next_node = int(action)

        # 1. Dinámica de transición
        next_time = self._transition(next_node)

        # 2. Condiciones de terminación (antes de actualizar el estado)
        terminated, truncated = self._check_termination(next_node, next_time)

        # 3. Recompensa del paso
        reward = self._compute_reward(next_node, next_time, terminated)

        # 4. Actualizar estado interno
        self._visited_set = self._visited_set | {next_node}
        self._current_node = next_node
        self._time_elapsed = next_time
        self._step_count += 1

        # 5. Observación y máscara del siguiente estado
        next_obs = self._build_obs()
        info = {"action_mask": self._get_action_mask()}

        return next_obs, float(reward), terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────────────
    # Separación arquitectónica: componentes internos de lógica de negocio
    # ─────────────────────────────────────────────────────────────────────────

    def _transition(self, next_node: int) -> float:
        """
        Dinámica de transición: calcula el tiempo acumulado tras el movimiento.

        Parámetros
        ----------
        next_node : int — nodo destino.

        Retorna
        -------
        float — nuevo tiempo total transcurrido en el episodio.
        """
        step_time = (
            self._time_matrix.iloc[self._current_node, next_node]
            if hasattr(self._time_matrix, "iloc")
            else float(self._time_matrix[self._current_node][next_node])
        )
        return self._time_elapsed + float(step_time)

    def _compute_reward(
        self, next_node: int, next_time: float, terminated: bool
    ) -> float:
        """
        Cálculo de recompensa del arco (current_node → next_node).

        Componentes:
        - Recompensa base estocástica o determinista del arco recorrido.
          En modo estocástico (STOCHASTIC_MODE=True) aplica ruido gaussiano
          que modela fluctuaciones de tarifa, cancelaciones y precio de diésel.
        - Bonificación por retorno exitoso: RETURN_SUCCESS_BONUS si el camión
          llega al nodo de inicio dentro del límite de tiempo.
        - Penalización por violación temporal: TIME_VIOLATION_PENALTY si se
          excede max_duration, ya sea al regresar o durante el trayecto.

        La penalización INCOMPLETE_PENALTY (episodio truncado sin retorno)
        se aplica externamente en training.py / tuning.py para preservar
        el comportamiento de memoria adicional del agente.

        Parámetros
        ----------
        next_node  : int   — nodo destino del movimiento.
        next_time  : float — tiempo acumulado tras el movimiento.
        terminated : bool  — True si el episodio termina en este paso.

        Retorna
        -------
        float — recompensa total del paso.
        """
        raw_reward = (
            self._reward_matrix_penalized.iloc[self._current_node, next_node]
            if hasattr(self._reward_matrix_penalized, "iloc")
            else float(self._reward_matrix_penalized[self._current_node][next_node])
        )

        # Recompensa base: estocástica o determinista
        if STOCHASTIC_MODE and self._noise_sigma > 0:
            step_reward = sample_stochastic_reward(
                raw_reward, self._noise_sigma, REWARD_SCALE_FACTOR
            )
        else:
            step_reward = float(raw_reward) / REWARD_SCALE_FACTOR

        # Bono / penalización terminal
        terminal_reward = 0.0
        if terminated:
            if next_node == self._start_node:
                # Ciclo cerrado: éxito o retorno fuera de tiempo
                terminal_reward = (
                    RETURN_SUCCESS_BONUS
                    if next_time <= self.max_duration
                    else TIME_VIOLATION_PENALTY
                )
            else:
                # Terminación forzada por violación del límite temporal en ruta
                terminal_reward = TIME_VIOLATION_PENALTY

        return step_reward + terminal_reward

    def _check_termination(self, next_node: int, next_time: float):
        """
        Evalúa las condiciones de fin de episodio.

        terminated : True si el camión cerró el ciclo (regresó al start_node)
                     o si el tiempo acumulado excede max_duration.
        truncated  : True si se alcanzó el límite de pasos (max_steps) sin
                     que ocurra terminación natural. Se computa con
                     ``step_count + 1`` porque step_count aún no fue incrementado.

        Parámetros
        ----------
        next_node : int   — nodo destino del movimiento en evaluación.
        next_time : float — tiempo acumulado si se realiza el movimiento.

        Retorna
        -------
        (terminated: bool, truncated: bool)
        """
        terminated = (
            next_node == self._start_node   # ciclo cerrado exitosamente
            or next_time > self.max_duration  # violación dura de tiempo
        )
        # truncated: límite de pasos alcanzado sin terminación natural
        truncated = (
            (not terminated)
            and (self._step_count >= self.max_steps - 1)
        )
        return terminated, truncated

    def _get_action_mask(self) -> np.ndarray:
        """
        Genera la máscara binaria de acciones válidas para el estado actual.

        Reglas de enmascaramiento:
        - 0 para self-loop (nodo actual del camión).
        - 0 para nodos intermedios ya visitados. El nodo de inicio (start_node)
          permanece disponible (máscara = 1) para que el agente pueda retornar
          en cualquier paso; no se fuerza el retorno para preservar el espacio
          de aprendizaje original (el agente aprende a retornar via reward shaping).
        - 1 para todos los demás nodos no visitados.

        Nota: la factibilidad temporal NO se aplica aquí (durante entrenamiento).
        El agente aprende a evitar callejones temporales mediante INCOMPLETE_PENALTY
        y TIME_VIOLATION_PENALTY. Aplicar la máscara de factibilidad en entrenamiento
        resulta demasiado restrictiva y aumenta el gap vs MIP. El lookahead temporal
        se aplica únicamente en el rollout greedy de evaluación (evaluation.py).

        Retorna
        -------
        np.ndarray[int8, shape=(num_nodes,)]
            1 = acción permitida, 0 = acción prohibida.
        """
        mask = np.ones(self.num_nodes, dtype=np.int8)

        remaining_arcs = self.max_steps - self._step_count

        # Si solo queda un arco, el único movimiento válido es volver al depot
        if remaining_arcs == 1:
            mask[:] = 0
            mask[self._start_node] = 1
            return mask

        # Self-loop siempre prohibido
        mask[self._current_node] = 0

        # Nodos intermedios ya visitados (start_node queda disponible para retorno)
        for v in self._visited_set:
            if v != self._start_node:
                mask[v] = 0

        return mask

    # ─────────────────────────────────────────────────────────────────────────
    # API pública auxiliar
    # ─────────────────────────────────────────────────────────────────────────

    def update_reward_matrix(self, reward_matrix_penalized) -> None:
        """Replace the reward matrix before the next episode.

        Call this before env.reset() to inject a new day's reward snapshot.
        Compatible with pd.DataFrame and np.ndarray (same as the constructor).
        """
        self._reward_matrix_penalized = reward_matrix_penalized

    def get_valid_actions(self) -> list:
        """
        Retorna la lista de índices de nodos con acción válida en el estado actual.

        Basado en la misma lógica de enmascaramiento de _get_action_mask():
        excluye self-loops y nodos intermedios visitados; incluye start_node
        como destino de retorno válido.

        Retorna
        -------
        list[int] — índices de acciones permitidas.
        """
        return [i for i, m in enumerate(self._get_action_mask()) if m == 1]

    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades de solo lectura del estado del episodio
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def start_node(self) -> int:
        """Nodo de inicio del episodio activo."""
        return self._start_node

    @property
    def current_node(self) -> int:
        """Posición actual del camión en el grafo."""
        return self._current_node

    @property
    def time_elapsed(self) -> float:
        """Tiempo acumulado (horas) en el episodio activo."""
        return self._time_elapsed

    @property
    def visited_set(self) -> frozenset:
        """Conjunto inmutable de nodos visitados en el episodio activo."""
        return frozenset(self._visited_set)

    # ─────────────────────────────────────────────────────────────────────────
    # Auxiliar interno
    # ─────────────────────────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """Construye el vector de observación float32 desde el estado actual."""
        return build_state(
            self._current_node,
            self._time_elapsed,
            self._visited_set,
            self._step_count,
            self.max_duration,
            self.max_steps,
            self.num_nodes,
        )
