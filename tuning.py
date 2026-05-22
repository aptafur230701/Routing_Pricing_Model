"""
tuning.py
=========
Optimización de hiperparámetros con Optuna.

Utiliza RoutingEnv (Gymnasium) como interfaz estándar de interacción.
El ciclo del episodio manual se reemplaza por la API de cinco valores de Gymnasium.

La conversión action_mask → invalid_actions se realiza internamente en este
módulo para no modificar la firma pública de DQNAgent_Optimized.act().

run_optuna(...)  →  dict de mejores hiperparámetros
"""

import random
import numpy as np
import torch
import optuna

from config import (
    STOCHASTIC_MODE,
    MAX_STEPS_PER_EPISODE,
    MAX_DURATION,
    INCOMPLETE_PENALTY,
    DEVICE,
)
from state import get_state_size
from agent import DQNAgent_Optimized
from routing_env import RoutingEnv


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares privados del módulo
# ─────────────────────────────────────────────────────────────────────────────

def _get_optuna_episodes(num_nodes: int) -> int:
    """Número de episodios de entrenamiento por trial según el tamaño del grafo."""
    if num_nodes <= 10:
        return 50 #luego cambiar a 5000
    if num_nodes <= 15:
        return 55 #luego cambiar a 5500
    if num_nodes <= 20:
        return 60 #luego cambiar a 6000
    return 75 #luego cambiar a 7500


def _mask_to_invalid(action_mask: np.ndarray) -> set:
    """
    Convierte la máscara binaria de Gymnasium al conjunto de acciones inválidas
    que espera DQNAgent_Optimized.act().

    Gymnasium: 1 = válido, 0 = inválido.
    Agente    : invalid_actions = conjunto de índices a bloquear.
    """
    return {int(i) for i, m in enumerate(action_mask) if m == 0}


def _run_trial_episode(
    agent: DQNAgent_Optimized,
    env: RoutingEnv,
    start_node: int,
    target_update_freq: int,
    total_steps: int,
) -> tuple:
    """
    Ejecuta un episodio de entrenamiento dentro de un trial de Optuna.

    Parámetros
    ----------
    agent              : DQNAgent_Optimized — agente en entrenamiento.
    env                : RoutingEnv — entorno Gymnasium.
    start_node         : int — nodo de inicio del episodio.
    target_update_freq : int — frecuencia de sincronización de la red target.
    total_steps        : int — contador global de pasos acumulados.

    Retorna
    -------
    (total_steps: int, episode_reward: float)
    """
    obs, info = env.reset(options={"start_node": start_node})
    episode_reward = 0.0
    truncated_ep = False

    for _ in range(env.max_steps):
        invalid_actions = _mask_to_invalid(info["action_mask"])
        action = agent.act(obs, invalid_actions=invalid_actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.remember(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_reward += reward
        total_steps += 1

        agent.decay_epsilon(total_steps)
        agent.replay(current_step=total_steps)
        if total_steps % target_update_freq == 0:
            agent.update_target_model()

        if terminated:
            break
        if truncated:
            truncated_ep = True
            break

    # Penalización por episodio incompleto (misma lógica que training.py)
    if truncated_ep:
        episode_reward += INCOMPLETE_PENALTY
        agent.remember(obs, int(obs[0]), INCOMPLETE_PENALTY, obs, True)

    return total_steps, episode_reward


def _run_validation_episode(
    agent: DQNAgent_Optimized,
    env: RoutingEnv,
    start_node: int,
) -> float:
    """
    Ejecuta un episodio de validación greedy (epsilon=0, sin actualización de pesos).

    Parámetros
    ----------
    agent      : DQNAgent_Optimized — agente con epsilon=0.
    env        : RoutingEnv — entorno Gymnasium.
    start_node : int — nodo de inicio del episodio de validación.

    Retorna
    -------
    float — recompensa acumulada del episodio.
    """
    obs, info = env.reset(options={"start_node": start_node})
    ep_reward = 0.0

    for _ in range(env.max_steps):
        invalid_actions = _mask_to_invalid(info["action_mask"])
        action = agent.act(obs, invalid_actions=invalid_actions)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        obs = next_obs

        if terminated or truncated:
            break

    return ep_reward


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de tuning
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna(
    time_matrix,
    reward_matrix_penalized,
    noise_sigma: float,
    num_nodes: int,
    epsilon_decay_steps: int,
    n_trials: int = 75,
) -> dict:
    """
    Ejecuta el estudio de Optuna y retorna los mejores hiperparámetros.

    El espacio de búsqueda (learning_rate, gamma, epsilon, buffer_size, etc.)
    y la arquitectura de redes (h1..h4) son idénticos al diseño original.
    NO se modifica DuelingQNetwork, ValueNetwork, PrioritizedReplayBuffer
    ni la lógica Double DQN.

    Parámetros
    ----------
    time_matrix             : pd.DataFrame | np.ndarray — tiempos entre nodos.
    reward_matrix_penalized : pd.DataFrame | np.ndarray — recompensas penalizadas.
    noise_sigma             : float — sigma del ruido estocástico.
    num_nodes               : int   — tamaño del grafo.
    epsilon_decay_steps     : int   — referencia para el espacio de búsqueda de epsilon.
    n_trials                : int   — número de trials de Optuna.

    Retorna
    -------
    dict — mejores hiperparámetros encontrados por Optuna.
    """
    state_size = get_state_size(num_nodes)

    def objective(trial):
        seed = 10
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Trial {trial.number} corriendo...")

        # ── Espacio de búsqueda (sin cambios respecto al diseño original) ──
        lr = trial.suggest_float("learning_rate", 1.7e-4, 2.5e-4, log=True)
        gamma = trial.suggest_float("gamma", 0.94, 0.96)
        eps_start = trial.suggest_float("epsilon_start", 0.6, 0.85)
        eps_end = trial.suggest_float("epsilon_end", 0.03, 0.07)
        eps_decay = trial.suggest_int(
            "epsilon_decay_steps",
            int(0.7 * epsilon_decay_steps),
            int(1.1 * epsilon_decay_steps),
        )
        base_buf = max(20_000, num_nodes ** 2 * 40)
        buffer_size = trial.suggest_int(
            "buffer_size", int(base_buf * 0.8), int(base_buf * 1.5)
        )
        batch_size = trial.suggest_int(
            "batch_size", num_nodes ** 2 // 3, num_nodes ** 2 // 2
        )
        tgt_update = trial.suggest_int("target_update_freq", 40, 60)
        grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)
        h1 = trial.suggest_int("hidden1", num_nodes ** 2, num_nodes ** 2 * 2)
        h2 = trial.suggest_int("hidden2", num_nodes ** 2 * 2, num_nodes ** 2 * 4)
        h3 = trial.suggest_int("hidden3", num_nodes ** 2, num_nodes ** 2 * 4)
        h4 = trial.suggest_int(
            "hidden4", int(num_nodes ** 2 * 0.8), int(num_nodes ** 2 * 1.2)
        )

        num_episodes = _get_optuna_episodes(num_nodes)
        total_steps_est = num_episodes * MAX_STEPS_PER_EPISODE

        # ── Agente del trial ──────────────────────────────────────
        agent = DQNAgent_Optimized(
            state_size=state_size,
            action_size=num_nodes,
            learning_rate=lr,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=DEVICE,
            num_nodes=num_nodes,
            total_training_steps=total_steps_est,
            epsilon_start=eps_start,
            epsilon_end=eps_end,
            epsilon_decay_steps=eps_decay,
            target_update_freq=tgt_update,
            h1=h1,
            h2=h2,
            h3=h3,
            h4=h4,
            grad_clip=grad_clip,
        )

        # ── Entorno Gymnasium del trial ───────────────────────────
        env = RoutingEnv(
            time_matrix=time_matrix,
            reward_matrix_penalized=reward_matrix_penalized,
            noise_sigma=noise_sigma,
            num_nodes=num_nodes,
            max_steps=MAX_STEPS_PER_EPISODE,
            max_duration=MAX_DURATION,
        )

        # ── Fase de entrenamiento del trial ───────────────────────
        total_steps = 0
        for ep in range(num_episodes):
            start_node = ep % num_nodes
            total_steps, _ = _run_trial_episode(
                agent, env, start_node, tgt_update, total_steps
            )

        # ── Fase de validación greedy (sin exploración) ───────────
        agent.epsilon = 0.0
        n_val = 30 if STOCHASTIC_MODE else 10
        val_rewards = [
            _run_validation_episode(agent, env, start_node=0)
            for _ in range(n_val)
        ]

        result = float(np.mean(val_rewards))
        print(f"Trial {trial.number + 1}/{n_trials} finalizado — val reward: {result:.2f}")
        return result

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Optuna mejor valor  : {study.best_value:.2f}")
    print(f"Optuna mejores params: {study.best_params}")
    return study.best_params
