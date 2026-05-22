"""
test_run.py
===========
Quick smoke-test: 5 nodes, 5 training episodes, no Optuna.
Verifies the full pipeline runs without crashes after the bug fixes.

Usage
-----
  python test_run.py
"""

import random
import numpy as np
import torch

from config import (
    SEED, DEVICE, MAX_STEPS_PER_EPISODE, MAX_DURATION,
    REWARD_SCALE_FACTOR, RETURN_SUCCESS_BONUS, TIME_VIOLATION_PENALTY,
    INCOMPLETE_PENALTY, STOCHASTIC_MODE,
)
from problem_data import load_matrices
from state import get_state_size, build_state
from agent import DQNAgent_Optimized
from problem_data import sample_stochastic_reward
from evaluation import generate_optimal_route

NUM_NODES     = 5
NUM_EPISODES  = 5


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_test():
    print(f"\n{'='*50}")
    print(f"  SMOKE TEST — {NUM_NODES} nodes, {NUM_EPISODES} episodes")
    print(f"{'='*50}")

    set_seeds(SEED)

    # ── 1. Load data ──────────────────────────────────────────
    time_matrix, reward_matrix, reward_matrix_penalized, noise_sigma = \
        load_matrices(NUM_NODES)

    print(f"  time_matrix shape   : {time_matrix.shape}")
    print(f"  reward_matrix shape : {reward_matrix.shape}")
    print(f"  noise_sigma         : {noise_sigma:.2f}")

    # ── 2. Build agent with simple default params ─────────────
    state_size = get_state_size(NUM_NODES)
    n2         = NUM_NODES ** 2

    best_params = {
        'learning_rate':       1e-3,
        'gamma':               0.95,
        'buffer_size':         500,
        'batch_size':          8,
        'epsilon_start':       1.0,
        'epsilon_end':         0.05,
        'epsilon_decay_steps': NUM_EPISODES * MAX_STEPS_PER_EPISODE,
        'target_update_freq':  10,
        'hidden1': n2, 'hidden2': n2 * 2,
        'hidden3': n2 * 2, 'hidden4': max(1, n2 // 2),
        'grad_clip': 1.0,
    }

    total_steps_est = NUM_EPISODES * MAX_STEPS_PER_EPISODE

    agent = DQNAgent_Optimized(
        state_size=state_size,
        action_size=NUM_NODES,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        buffer_size=best_params['buffer_size'],
        batch_size=best_params['batch_size'],
        device=DEVICE,
        num_nodes=NUM_NODES,
        total_training_steps=total_steps_est,
        epsilon_start=best_params['epsilon_start'],
        epsilon_end=best_params['epsilon_end'],
        epsilon_decay_steps=best_params['epsilon_decay_steps'],
        target_update_freq=best_params['target_update_freq'],
        h1=best_params['hidden1'], h2=best_params['hidden2'],
        h3=best_params['hidden3'], h4=best_params['hidden4'],
        grad_clip=best_params['grad_clip'],
    )
    print(f"  Agent built OK | state_size={state_size} | action_size={NUM_NODES}")

    # ── 3. Mini training loop ─────────────────────────────────
    print(f"\n  Training {NUM_EPISODES} episodes...")
    episode_rewards = []
    total_steps     = 0

    for episode in range(NUM_EPISODES):
        start_node   = episode % NUM_NODES
        current_node = start_node
        time_elapsed = 0.0
        visited_set  = {start_node}
        state        = build_state(current_node, time_elapsed, visited_set,
                                   0, MAX_DURATION, MAX_STEPS_PER_EPISODE, NUM_NODES)
        ep_reward    = 0.0
        done         = False

        for step in range(MAX_STEPS_PER_EPISODE):
            # BUG 1 FIX: pass invalid_actions so revisits are blocked during training
            action    = agent.act(state, invalid_actions=visited_set - {start_node})
            next_node = action

            raw_reward = (reward_matrix_penalized.iloc[current_node, next_node]
                          if hasattr(reward_matrix_penalized, 'iloc')
                          else reward_matrix_penalized[current_node][next_node])
            step_reward = (sample_stochastic_reward(raw_reward, noise_sigma, REWARD_SCALE_FACTOR)
                           if STOCHASTIC_MODE else raw_reward / REWARD_SCALE_FACTOR)

            step_time  = (time_matrix.iloc[current_node, next_node]
                          if hasattr(time_matrix, 'iloc')
                          else time_matrix[current_node][next_node])
            next_time  = time_elapsed + step_time
            visited_next = visited_set | {next_node}
            next_state = build_state(next_node, next_time, visited_next,
                                     step + 1, MAX_DURATION, MAX_STEPS_PER_EPISODE, NUM_NODES)

            terminal_reward = 0.0
            if next_node == start_node:
                terminal_reward = RETURN_SUCCESS_BONUS if next_time <= MAX_DURATION else TIME_VIOLATION_PENALTY
                done = True
            elif next_time > MAX_DURATION:
                terminal_reward = TIME_VIOLATION_PENALTY
                done = True

            agent.remember(state, action, step_reward + terminal_reward, next_state, done)

            state        = next_state
            current_node = next_node
            time_elapsed = next_time
            visited_set  = visited_next
            ep_reward   += step_reward + terminal_reward
            total_steps += 1

            agent.decay_epsilon(total_steps)
            agent.replay(current_step=total_steps)
            if total_steps % agent.target_update_freq == 0:
                agent.update_target_model()
            if done:
                break

        # BUG 2 FIX: penalise episodes where agent never returned home
        if not done:
            ep_reward += INCOMPLETE_PENALTY
            agent.remember(state, int(state[0]), INCOMPLETE_PENALTY, state, True)

        episode_rewards.append(ep_reward)
        print(f"    ep {episode+1}/{NUM_EPISODES} | start={start_node} | "
              f"reward={ep_reward:.3f} | eps={agent.epsilon:.3f}")

    print(f"  Training complete. avg_reward={np.mean(episode_rewards):.3f}")

    # ── 4. Greedy evaluation ──────────────────────────────────
    print(f"\n  Greedy evaluation (epsilon=0)...")
    for s in range(NUM_NODES):
        route, reward, duration = generate_optimal_route(
            agent, s, time_matrix, reward_matrix_penalized, NUM_NODES)
        status = "OK" if route is not None else "NO ROUTE"
        print(f"    start={s} | {status} | route={route} | "
              f"reward={reward:.1f} | duration={duration:.1f} min")

    print(f"\n{'='*50}")
    print("  SMOKE TEST PASSED — no crashes detected.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_test()
