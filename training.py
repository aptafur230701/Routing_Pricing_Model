"""
training.py
===========
Full training loop with balanced start-node sampling.

run_training(...)  →  (trained DQNAgent_Optimized, episode_rewards, episode_losses)
"""

import os
import random
from collections import deque
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter

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


def _run_greedy_episode(
    agent, time_matrix, reward_matrix_penalized, noise_sigma, num_nodes
) -> tuple:
    """One greedy episode from node 0 with epsilon=0.
    Returns (reward, valid_cycle_flag, duration)."""
    saved_eps = agent.epsilon
    agent.epsilon = 0.0

    start_node   = 0
    current_node = start_node
    time_elapsed = 0.0
    visited_set  = {start_node}
    state        = build_state(current_node, time_elapsed, visited_set,
                               0, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)
    ep_reward   = 0.0
    done        = False
    valid_cycle = False

    for step in range(MAX_STEPS_PER_EPISODE):
        action    = agent.act(state, invalid_actions=visited_set - {start_node})
        next_node = action

        raw_reward  = (reward_matrix_penalized.iloc[current_node, next_node]
                       if hasattr(reward_matrix_penalized, 'iloc')
                       else reward_matrix_penalized[current_node][next_node])
        step_reward = (sample_stochastic_reward(raw_reward, noise_sigma, REWARD_SCALE_FACTOR)
                       if STOCHASTIC_MODE else raw_reward / REWARD_SCALE_FACTOR)

        step_time   = (time_matrix.iloc[current_node, next_node]
                       if hasattr(time_matrix, 'iloc')
                       else time_matrix[current_node][next_node])
        next_time    = time_elapsed + step_time
        visited_next = visited_set | {next_node}
        next_state   = build_state(next_node, next_time, visited_next,
                                   step + 1, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)

        terminal_reward = 0.0
        done = False
        if next_node == start_node:
            if next_time <= MAX_DURATION:
                terminal_reward = RETURN_SUCCESS_BONUS
                valid_cycle     = True
            else:
                terminal_reward = TIME_VIOLATION_PENALTY
            done = True
        elif next_time > MAX_DURATION:
            terminal_reward = TIME_VIOLATION_PENALTY
            done = True

        ep_reward   += step_reward + terminal_reward
        state        = next_state
        current_node = next_node
        time_elapsed = next_time
        visited_set  = visited_next

        if done:
            break

    if not done:
        ep_reward += INCOMPLETE_PENALTY

    agent.epsilon = saved_eps
    return ep_reward, float(valid_cycle), time_elapsed


def run_training(
    best_params:             dict,
    time_matrix,
    reward_matrix_penalized,
    noise_sigma:             float,
    num_nodes:               int,
    log_dir:                 str = "runs",
    eval_freq:               int = 200,
    early_stop_patience:     int = 2000,
    n_greedy_eval:           int = 10,
) -> tuple:
    """Build a fresh agent with best_params and train it fully.

    Returns
    -------
    agent           : DQNAgent_Optimized  (trained)
    episode_rewards : list[float]
    episode_losses  : list[float]
    """
    state_size        = get_state_size(num_nodes)
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

    run_name        = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{num_nodes}nodes"
    writer          = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    best_model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "best_model.pt"
    )

    episode_rewards     = []
    episode_losses      = []
    total_steps         = 0
    node_ep_counts      = {n: 0 for n in range(num_nodes)}
    rolling_rewards     = deque(maxlen=100)
    best_eval_reward    = -float('inf')
    last_improvement_ep = 0

    for episode in range(num_episodes):
        start_node = _select_start_node(episode, num_episodes, num_nodes, node_ep_counts)
        node_ep_counts[start_node] += 1

        current_node  = start_node
        time_elapsed  = 0.0
        visited_set   = {start_node}
        state         = build_state(current_node, time_elapsed, visited_set,
                                    0, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)
        ep_reward     = 0.0
        ep_q_loss_sum = 0.0
        ep_v_loss_sum = 0.0
        ep_grad_sum   = 0.0
        replay_count  = 0
        steps_in_ep   = 0
        done          = False
        valid_cycle   = False

        for step in range(MAX_STEPS_PER_EPISODE):
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
            next_time    = time_elapsed + step_time
            visited_next = visited_set | {next_node}
            next_state   = build_state(next_node, next_time, visited_next,
                                       step + 1, MAX_DURATION, MAX_STEPS_PER_EPISODE, num_nodes)

            terminal_reward = 0.0
            done            = False
            if next_node == start_node:
                if next_time <= MAX_DURATION:
                    terminal_reward = RETURN_SUCCESS_BONUS
                    valid_cycle     = True
                else:
                    terminal_reward = TIME_VIOLATION_PENALTY
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
            steps_in_ep += 1
            total_steps += 1

            agent.decay_epsilon(total_steps)
            q_loss, v_loss, grad_norm = agent.replay(current_step=total_steps)
            if q_loss > 0:
                ep_q_loss_sum += q_loss
                ep_v_loss_sum += v_loss
                ep_grad_sum   += grad_norm
                replay_count  += 1
            if total_steps % agent.target_update_freq == 0:
                agent.update_target_model()
            if done:
                break

        if not done:
            ep_reward += INCOMPLETE_PENALTY
            agent.remember(state, int(state[0]), INCOMPLETE_PENALTY, state, True)

        episode_rewards.append(ep_reward)
        rolling_rewards.append(ep_reward)
        avg_q_loss = ep_q_loss_sum / replay_count if replay_count > 0 else 0.0
        avg_v_loss = ep_v_loss_sum / replay_count if replay_count > 0 else 0.0
        avg_grad   = ep_grad_sum   / replay_count if replay_count > 0 else 0.0
        episode_losses.append(avg_q_loss)

        # TensorBoard — training metrics (every episode)
        writer.add_scalar("train/reward",            ep_reward,                        episode)
        writer.add_scalar("train/reward_rolling100", float(np.mean(rolling_rewards)),  episode)
        writer.add_scalar("train/q_loss",            avg_q_loss,                       episode)
        writer.add_scalar("train/v_loss",            avg_v_loss,                       episode)
        writer.add_scalar("train/valid_route_rate",  float(valid_cycle),               episode)
        writer.add_scalar("train/steps_per_episode", steps_in_ep,                      episode)
        writer.add_scalar("agent/epsilon",           agent.epsilon,                    episode)
        writer.add_scalar("agent/learning_rate",
                          agent.optimizer.param_groups[0]['lr'],                       episode)
        writer.add_scalar("agent/grad_norm",         avg_grad,                         episode)

        buf_size, buf_beta, buf_mean_prio = agent.get_buffer_stats(total_steps)
        writer.add_scalar("buffer/size",          buf_size,      episode)
        writer.add_scalar("buffer/beta",          buf_beta,      episode)
        writer.add_scalar("buffer/mean_priority", buf_mean_prio, episode)

        # Greedy evaluation every eval_freq episodes
        if (episode + 1) % eval_freq == 0:
            g_rewards, g_valids, g_durs = [], [], []
            for _ in range(n_greedy_eval):
                g_r, g_v, g_d = _run_greedy_episode(
                    agent, time_matrix, reward_matrix_penalized, noise_sigma, num_nodes)
                g_rewards.append(g_r)
                g_valids.append(g_v)
                g_durs.append(g_d)
            mean_g_reward = float(np.mean(g_rewards))
            mean_g_valid  = float(np.mean(g_valids))
            mean_g_dur    = float(np.mean(g_durs))
            writer.add_scalar("eval/greedy_reward",   mean_g_reward, episode)
            writer.add_scalar("eval/greedy_valid",    mean_g_valid,  episode)
            writer.add_scalar("eval/greedy_duration", mean_g_dur,    episode)

            if mean_g_reward > best_eval_reward:
                best_eval_reward    = mean_g_reward
                last_improvement_ep = episode
                agent.save(best_model_path)
                print(f"  ** New best eval reward: {best_eval_reward:.3f} "
                      f"→ saved {best_model_path}")

            if episode >= early_stop_patience and \
                    (episode - last_improvement_ep) >= early_stop_patience:
                print(f"\nEarly stopping at episode {episode + 1}: "
                      f"no improvement for {early_stop_patience} episodes.")
                break

        log_freq = max(1, num_episodes // 10)
        if (episode + 1) % log_freq == 0:
            print(f"  ep {episode+1:>6}/{num_episodes} | "
                  f"steps {steps_in_ep} | reward {ep_reward:6.1f} | "
                  f"loss {avg_q_loss:.4f} | eps {agent.epsilon:.3f}")

    writer.close()
    print("Training complete.")
    return agent, episode_rewards, episode_losses