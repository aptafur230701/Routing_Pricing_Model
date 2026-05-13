"""
training.py
===========
Full training loop with balanced start-node sampling.

run_training(...)  →  (trained DQNAgent_Optimized, episode_rewards, episode_losses)
"""

import random
import numpy as np

from config import (
    STOCHASTIC_MODE, MAX_STEPS_PER_EPISODE, MAX_DURATION,
    REWARD_SCALE_FACTOR, RETURN_SUCCESS_BONUS, TIME_VIOLATION_PENALTY,
    INCOMPLETE_PENALTY, DEVICE, get_episodes_per_node, get_buffer_size,
)
from state import get_state_size, build_state
from agent import DQNAgent_Optimized
from environment import sample_stochastic_reward


def _select_start_node(episode: int, total_episodes: int,
                        num_nodes: int, episode_counts: dict) -> int:
    """Random for first 50 % of training, then balanced across nodes."""
    if episode < int(0.5 * total_episodes):
        return random.randint(0, num_nodes - 1)
    min_count  = min(episode_counts.values())
    candidates = [n for n, c in episode_counts.items() if c == min_count]
    return random.choice(candidates)


def run_training(
    best_params:             dict,
    time_matrix,
    reward_matrix_penalized,
    noise_sigma:             float,
    num_nodes:               int,
) -> tuple:
    """Build a fresh agent with best_params and train it fully.

    Returns
    -------
    agent           : DQNAgent_Optimized  (trained)
    episode_rewards : list[float]
    episode_losses  : list[float]
    """
    state_size   = get_state_size(num_nodes)
    episodes_per_node = get_episodes_per_node(num_nodes)
    num_episodes      = episodes_per_node * num_nodes
    total_steps_est   = num_episodes * MAX_STEPS_PER_EPISODE

    agent = DQNAgent_Optimized(
        state_size=state_size,
        action_size=num_nodes,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        buffer_size=best_params['buffer_size'],
        batch_size=best_params['batch_size'],
        device=DEVICE,
        num_nodes=num_nodes,
        total_training_steps=total_steps_est,
        epsilon_start=best_params['epsilon_start'],
        epsilon_end=best_params['epsilon_end'],
        epsilon_decay_steps=best_params['epsilon_decay_steps'],
        target_update_freq=best_params['target_update_freq'],
        h1=best_params['hidden1'], h2=best_params['hidden2'],
        h3=best_params['hidden3'], h4=best_params['hidden4'],
        grad_clip=best_params['grad_clip'],
    )

    print(f"\n--- Full Training: {num_episodes} episodes "
          f"({episodes_per_node} per node × {num_nodes} nodes) ---")

    episode_rewards  = []
    episode_losses   = []
    total_steps      = 0
    node_ep_counts   = {n: 0 for n in range(num_nodes)}

    for episode in range(num_episodes):
        start_node   = _select_start_node(episode, num_episodes, num_nodes, node_ep_counts)
        node_ep_counts[start_node] += 1

        current_node  = start_node
        time_elapsed  = 0.0
        visited_set   = {start_node}
        state         = build_state(current_node, time_elapsed, visited_set,
                                    0, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)
        ep_reward     = 0.0
        ep_loss_sum   = 0.0
        steps_in_ep   = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            # Block intermediate nodes already visited; allow return to start_node
            action    = agent.act(state, invalid_actions=visited_set - {start_node})
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
            visited_next = visited_set | {next_node}
            next_state  = build_state(next_node, next_time, visited_next,
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

            state        = next_state
            current_node = next_node
            time_elapsed = next_time
            visited_set  = visited_next

            # Accumulate episode reward (terminal already included in memory entry)
            ep_reward  += step_reward + terminal_reward
            steps_in_ep += 1
            total_steps += 1

            agent.decay_epsilon(total_steps)
            loss = agent.replay(current_step=total_steps)
            if loss > 0:
                ep_loss_sum += loss
            if total_steps % agent.target_update_freq == 0:
                agent.update_target_model()
            if done:
                break

        # Penalise episodes where the agent never returned to start_node
        if not done:
            ep_reward += INCOMPLETE_PENALTY
            agent.remember(state, int(state[0]), INCOMPLETE_PENALTY, state, True)

        episode_rewards.append(ep_reward)
        avg_loss = ep_loss_sum / steps_in_ep if steps_in_ep > 0 else 0.0
        episode_losses.append(avg_loss)

        log_freq = max(1, num_episodes // 10)
        if (episode + 1) % log_freq == 0:
            print(f"  ep {episode+1:>6}/{num_episodes} | "
                  f"steps {steps_in_ep} | reward {ep_reward:6.1f} | "
                  f"loss {avg_loss:.4f} | eps {agent.epsilon:.3f}")

    print("Training complete.")
    return agent, episode_rewards, episode_losses
