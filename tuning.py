"""
tuning.py
=========
Optuna hyperparameter optimisation.

run_optuna(...)  →  dict of best hyperparameters
"""

import random
from unittest import result
import numpy as np
import torch
import optuna

from config import (
    STOCHASTIC_MODE, MAX_STEPS_PER_EPISODE,
    MAX_DURATION, REWARD_SCALE_FACTOR,
    RETURN_SUCCESS_BONUS, TIME_VIOLATION_PENALTY,
    DEVICE,
)
from state import get_state_size, build_state
from agent import DQNAgent_Optimized
from environment import sample_stochastic_reward


def _get_optuna_episodes(num_nodes: int) -> int:
    if num_nodes <= 10:  return 5000
    if num_nodes <= 15:  return 5500
    if num_nodes <= 20:  return 6000
    return 7500


def _run_trial_episode(agent, start_node, time_matrix, reward_matrix_penalized,
                        noise_sigma, num_nodes, state_size, epsilon_decay_steps,
                        target_update_freq, total_steps):
    """One training episode inside an Optuna trial. Returns (total_steps, episode_reward)."""
    current_node  = start_node
    time_elapsed  = 0.0
    visited_set   = {start_node}
    state         = build_state(current_node, time_elapsed, visited_set, 0,
                                MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)
    episode_reward = 0.0

    for step in range(MAX_STEPS_PER_EPISODE):
        # Block already-visited intermediate nodes; allow return to start_node
        action    = agent.act(state, invalid_actions=visited_set - {start_node})
        next_node = action

        raw_reward = (reward_matrix_penalized.iloc[current_node, next_node]
                      if hasattr(reward_matrix_penalized, 'iloc')
                      else reward_matrix_penalized[current_node][next_node])

        step_reward = (sample_stochastic_reward(raw_reward, noise_sigma, REWARD_SCALE_FACTOR)
                       if STOCHASTIC_MODE
                       else raw_reward / REWARD_SCALE_FACTOR)

        step_time        = (time_matrix.iloc[current_node, next_node]
                            if hasattr(time_matrix, 'iloc')
                            else time_matrix[current_node][next_node])
        next_time        = time_elapsed + step_time
        visited_next     = visited_set | {next_node}
        next_state       = build_state(next_node, next_time, visited_next,
                                       step + 1, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)

        terminal_reward = 0.0
        done            = False
        if next_node == start_node:
            terminal_reward = RETURN_SUCCESS_BONUS if next_time <= MAX_DURATION else TIME_VIOLATION_PENALTY
            done = True
        elif next_time > MAX_DURATION:
            terminal_reward = TIME_VIOLATION_PENALTY
            done = True

        agent.remember(state, action, step_reward + terminal_reward, next_state, done)

        state         = next_state
        current_node  = next_node
        time_elapsed  = next_time
        visited_set   = visited_next
        episode_reward += step_reward
        total_steps   += 1

        agent.decay_epsilon(total_steps)
        agent.replay(current_step=total_steps)
        if total_steps % target_update_freq == 0:
            agent.update_target_model()
        if done:
            episode_reward += terminal_reward
            break

    return total_steps, episode_reward


def run_optuna(
    time_matrix,
    reward_matrix_penalized,
    noise_sigma:  float,
    num_nodes:    int,
    epsilon_decay_steps: int,
    n_trials:     int = 75,
) -> dict:
    """Run Optuna study and return best hyperparameters."""

    state_size = get_state_size(num_nodes)

    def objective(trial):
        seed = 10
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        print(f"Trial {trial.number} running...")

        lr                  = trial.suggest_float('learning_rate', 1.7e-4, 2.5e-4, log=True)
        gamma               = trial.suggest_float('gamma', 0.94, 0.96)
        eps_start           = trial.suggest_float('epsilon_start', 0.6, 0.85)
        eps_end             = trial.suggest_float('epsilon_end', 0.03, 0.07)
        eps_decay           = trial.suggest_int('epsilon_decay_steps',
                                int(0.7 * epsilon_decay_steps),
                                int(1.1 * epsilon_decay_steps))
        base_buf            = max(20_000, num_nodes ** 2 * 40)
        buffer_size         = trial.suggest_int('buffer_size', int(base_buf * 0.8), int(base_buf * 1.5))
        batch_size          = trial.suggest_int('batch_size', num_nodes**2 // 3, num_nodes**2 // 2)
        tgt_update          = trial.suggest_int('target_update_freq', 40, 60)
        grad_clip           = trial.suggest_float('grad_clip', 0.5, 5.0)
        h1 = trial.suggest_int('hidden1', num_nodes**2,     num_nodes**2 * 2)
        h2 = trial.suggest_int('hidden2', num_nodes**2 * 2, num_nodes**2 * 4)
        h3 = trial.suggest_int('hidden3', num_nodes**2,     num_nodes**2 * 4)
        h4 = trial.suggest_int('hidden4', int(num_nodes**2 * 0.8), int(num_nodes**2 * 1.2))

        num_episodes        = _get_optuna_episodes(num_nodes)
        total_steps_est     = num_episodes * MAX_STEPS_PER_EPISODE

        agent = DQNAgent_Optimized(
            state_size=state_size, action_size=num_nodes,
            learning_rate=lr, gamma=gamma,
            buffer_size=buffer_size, batch_size=batch_size,
            device=DEVICE, num_nodes=num_nodes,
            total_training_steps=total_steps_est,
            epsilon_start=eps_start, epsilon_end=eps_end,
            epsilon_decay_steps=eps_decay,
            target_update_freq=tgt_update,
            h1=h1, h2=h2, h3=h3, h4=h4, grad_clip=grad_clip,
        )

        total_steps = 0
        for ep in range(num_episodes):
            start_node  = ep % num_nodes
            total_steps, _ = _run_trial_episode(
                agent, start_node, time_matrix, reward_matrix_penalized,
                noise_sigma, num_nodes, state_size, eps_decay,
                tgt_update, total_steps)

        # Validation (greedy, no exploration)
        agent.epsilon = 0.0
        n_val         = 30 if STOCHASTIC_MODE else 10
        val_rewards   = []
        for _ in range(n_val):
            current_node = 0; time_elapsed = 0.0; visited_set = {0}
            state = build_state(0, 0.0, {0}, 0, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)
            ep_r  = 0.0
            for step in range(MAX_STEPS_PER_EPISODE):
                action    = agent.act(state, invalid_actions=visited_set - {0})
                next_node = action
                raw_reward = (reward_matrix_penalized.iloc[current_node, next_node]
                              if hasattr(reward_matrix_penalized, 'iloc')
                              else reward_matrix_penalized[current_node][next_node])
                step_reward = (sample_stochastic_reward(raw_reward, noise_sigma, REWARD_SCALE_FACTOR)
                               if STOCHASTIC_MODE else raw_reward / REWARD_SCALE_FACTOR)
                step_time   = (time_matrix.iloc[current_node, next_node]
                               if hasattr(time_matrix, 'iloc')
                               else time_matrix[current_node][next_node])
                next_time   = time_elapsed + step_time
                visited_set = visited_set | {next_node}
                next_state  = build_state(next_node, next_time, visited_set,
                                          step + 1, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)
                terminal_reward = 0.0; done = False
                if next_node == 0:
                    terminal_reward = RETURN_SUCCESS_BONUS if next_time <= MAX_DURATION else TIME_VIOLATION_PENALTY
                    done = True
                elif next_time > MAX_DURATION:
                    terminal_reward = TIME_VIOLATION_PENALTY; done = True
                ep_r += step_reward
                if done:
                    ep_r += terminal_reward; break
                state = next_state; current_node = next_node; time_elapsed = next_time
            val_rewards.append(ep_r)

            
        result = float(np.mean(val_rewards))
        print(f"Trial {trial.number + 1}/75 finished — val reward: {result:.2f}")
        return result


    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print(f"Optuna best value : {study.best_value:.2f}")
    print(f"Optuna best params: {study.best_params}")
    return study.best_params
