"""
training.py
===========
Bucle de entrenamiento completo con muestreo balanceado de nodos de inicio.

Utiliza RoutingEnv (Gymnasium) como interfaz estándar de interacción con el entorno.
El ciclo manual que antes vivía aquí se reemplaza por la API de cinco valores:

    obs, info                               = env.reset(options={"start_node": k})
    next_obs, reward, terminated, truncated, info = env.step(action)

La máscara de acciones de Gymnasium (info["action_mask"]) se convierte
internamente al formato de conjunto inválido que espera DQNAgent_Optimized.act(),
sin modificar la firma pública del agente.

run_training(...)  →  (DQNAgent_Optimized entrenado, episode_rewards, episode_losses)
"""

import random
import numpy as np

from config import (
    MAX_STEPS_PER_EPISODE,
    MAX_DURATION,
    INCOMPLETE_PENALTY,
    DEVICE,
    get_episodes_per_node,
)
from state import get_state_size
from agent import DQNAgent_Optimized
from routing_env import RoutingEnv
from problem_data import build_day_matrices


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares privados del módulo
# ─────────────────────────────────────────────────────────────────────────────

def _select_start_node(
    episode: int,
    total_episodes: int,
    num_nodes: int,
    episode_counts: dict,
) -> int:
    """
    Política de selección de nodo de inicio.

    Primero 50 % del entrenamiento: aleatorio uniforme.
    Segundo 50 %: balanceo — elige el nodo con menos episodios acumulados.
    """
    if episode < int(0.5 * total_episodes):
        return random.randint(0, num_nodes - 1)
    min_count = min(episode_counts.values())
    candidates = [n for n, c in episode_counts.items() if c == min_count]
    return random.choice(candidates)


def _mask_to_invalid(action_mask: np.ndarray) -> set:
    """
    Convierte la máscara binaria de Gymnasium al conjunto de acciones inválidas
    que espera DQNAgent_Optimized.act().

    Gymnasium: 1 = válido, 0 = inválido.
    Agente    : invalid_actions = conjunto de índices a bloquear.

    El agente añade internamente current_node al conjunto de bloqueados,
    por lo que incluirlo aquí (mask[current_node] = 0) es redundante pero
    inofensivo y mantiene la máscara semánticamente completa.
    """
    return {int(i) for i, m in enumerate(action_mask) if m == 0}


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    best_params:  dict,
    time_matrix,
    rate_stack:   np.ndarray,
    loads_stack:  np.ndarray,
    distance_arr: np.ndarray,
    diesel_arr:   np.ndarray,
    noise_sigma:  float,
    num_nodes:    int,
) -> tuple:
    """
    Construye un agente DQN nuevo con los hiperparámetros óptimos y lo entrena.

    En cada episodio se samplea un día aleatorio de los últimos 90 días para
    que el agente aprenda de variabilidad histórica en lugar de memorizar una
    distribución de recompensas única.

    Parámetros
    ----------
    best_params  : dict — hiperparámetros de Optuna.
    time_matrix  : pd.DataFrame — tiempos entre nodos.
    rate_stack   : np.ndarray [num_days, num_nodes, num_nodes] — tarifas históricas.
    loads_stack  : np.ndarray [num_days, num_nodes, num_nodes] — cargas históricas.
    distance_arr : np.ndarray [num_nodes, num_nodes] — distancias entre nodos.
    diesel_arr   : np.ndarray [num_nodes, num_nodes] — precios de combustible.
    noise_sigma  : float — sigma del ruido estocástico.
    num_nodes    : int  — tamaño del grafo.

    Retorna
    -------
    agent           : DQNAgent_Optimized entrenado.
    episode_rewards : list[float] — recompensa acumulada por episodio.
    episode_losses  : list[float] — pérdida promedio por episodio.
    """
    state_size        = get_state_size(num_nodes)
    episodes_per_node = get_episodes_per_node(num_nodes)
    num_episodes      = episodes_per_node * num_nodes
    total_steps_est   = num_episodes * MAX_STEPS_PER_EPISODE
    num_days          = rate_stack.shape[0]

    # ── Construcción del agente ───────────────────────────────────
    agent = DQNAgent_Optimized(
        state_size=state_size,
        action_size=num_nodes,
        learning_rate=best_params["learning_rate"],
        gamma=best_params["gamma"],
        buffer_size=best_params["buffer_size"],
        batch_size=best_params["batch_size"],
        device=DEVICE,
        num_nodes=num_nodes,
        total_training_steps=total_steps_est,
        epsilon_start=best_params["epsilon_start"],
        epsilon_end=best_params["epsilon_end"],
        epsilon_decay_steps=best_params["epsilon_decay_steps"],
        target_update_freq=best_params["target_update_freq"],
        h1=best_params["hidden1"],
        h2=best_params["hidden2"],
        h3=best_params["hidden3"],
        h4=best_params["hidden4"],
        grad_clip=best_params["grad_clip"],
    )

    # ── Construcción del entorno Gymnasium (día 0 como placeholder) ─
    _, reward_matrix_penalized_init = build_day_matrices(
        rate_stack[0], loads_stack[0], distance_arr, diesel_arr
    )
    env = RoutingEnv(
        time_matrix=time_matrix,
        reward_matrix_penalized=reward_matrix_penalized_init,
        noise_sigma=noise_sigma,
        num_nodes=num_nodes,
        max_steps=MAX_STEPS_PER_EPISODE,
        max_duration=MAX_DURATION,
    )

    print(
        f"\n--- Entrenamiento completo: {num_episodes} episodios "
        f"({episodes_per_node} por nodo × {num_nodes} nodos) ---"
    )

    episode_rewards = []
    episode_losses = []
    total_steps = 0
    node_ep_counts = {n: 0 for n in range(num_nodes)}

    # ── Bucle principal de entrenamiento ─────────────────────────
    for episode in range(num_episodes):
        start_node = _select_start_node(
            episode, num_episodes, num_nodes, node_ep_counts
        )
        node_ep_counts[start_node] += 1

        # Samplear un día aleatorio e inyectar su matriz de recompensas
        day_idx = np.random.randint(0, num_days)
        _, rm_pen = build_day_matrices(
            rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
        )
        env.update_reward_matrix(rm_pen)

        # Inicializar episodio vía API estándar de Gymnasium
        obs, info = env.reset(options={"start_node": start_node})

        ep_reward = 0.0
        ep_loss_sum = 0.0
        steps_in_ep = 0
        terminated_ep = False
        truncated_ep = False

        for _ in range(MAX_STEPS_PER_EPISODE):
            # Convertir action_mask de Gym → invalid_actions del agente
            invalid_actions = _mask_to_invalid(info["action_mask"])
            action = agent.act(obs, invalid_actions=invalid_actions)

            # Paso en el entorno: 5-tupla estándar de Gymnasium
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Almacenar experiencia en el replay buffer
            agent.remember(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            steps_in_ep += 1
            total_steps += 1

            # Decaimiento de epsilon y actualización de la red
            agent.decay_epsilon(total_steps)
            loss = agent.replay(current_step=total_steps)
            if loss > 0:
                ep_loss_sum += loss
            if total_steps % agent.target_update_freq == 0:
                agent.update_target_model()

            if terminated:
                terminated_ep = True
                break
            if truncated:
                truncated_ep = True
                break

        # Penalizar episodios en los que el agente no retornó al nodo de inicio.
        # Equivalente al comportamiento original: experiencia sintética adicional
        # con INCOMPLETE_PENALTY para reforzar que quedar bloqueado es negativo.
        if truncated_ep:
            ep_reward += INCOMPLETE_PENALTY
            agent.remember(obs, int(obs[0]), INCOMPLETE_PENALTY, obs, True)

        episode_rewards.append(ep_reward)
        avg_loss = ep_loss_sum / steps_in_ep if steps_in_ep > 0 else 0.0
        episode_losses.append(avg_loss)

        log_freq = max(1, num_episodes // 10)
        if (episode + 1) % log_freq == 0:
            print(
                f"  ep {episode+1:>6}/{num_episodes} | "
                f"steps {steps_in_ep} | reward {ep_reward:6.1f} | "
                f"loss {avg_loss:.4f} | eps {agent.epsilon:.3f}"
            )

    print("Entrenamiento completo.")
    return agent, episode_rewards, episode_losses
