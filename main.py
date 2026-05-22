"""
main.py
=======
Entry point. Orchestrates the full pipeline:

  1. Load data          (environment.py)
  2. Optuna tuning      (tuning.py)
  3. Full training      (training.py)
  4. Evaluation         (evaluation.py)
  5. Save results       (evaluation.py)

Usage
-----
  python main.py
"""

import os
import random
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

from config import (
    SEED, DEVICE, STOCHASTIC_MODE, NOISE_FRACTION,
    MAX_STEPS_PER_EPISODE, REWARD_SCALE_FACTOR,
    get_episodes_per_node, get_buffer_size,
)
from state import get_state_size
from problem_data import load_matrices
from tuning import run_optuna
from training import run_training
from evaluation import (
    run_solver_comparison, save_results, plot_diagnostics,
)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    cwd          = os.path.dirname(os.path.abspath(__file__))
    summary_rows = []

    for NUM_NODES in [10]:
        print(f"\n{'='*60}")
        print(f"  Stochastic Optimised DRL — {NUM_NODES} nodes")
        print(f"{'='*60}")

        set_seeds(SEED)

        # 1. Data
        time_matrix, reward_matrix, reward_matrix_penalized, noise_sigma = \
            load_matrices(NUM_NODES)

        state_size         = get_state_size(NUM_NODES)
        episodes_per_node  = get_episodes_per_node(NUM_NODES)
        num_episodes       = episodes_per_node * NUM_NODES
        epsilon_decay_steps = num_episodes * MAX_STEPS_PER_EPISODE

        print(f"Device          : {DEVICE}")
        print(f"State size      : {state_size}  (2 + {NUM_NODES} visited + 2 step)")
        print(f"Training episodes: {num_episodes}")

        # 2. Optuna
        print("\n--- Optuna (75 trials) ---")
        best_params = run_optuna(
            time_matrix, reward_matrix_penalized,
            noise_sigma, NUM_NODES, epsilon_decay_steps,
            n_trials=75,
        )

        # 3. Full training
        set_seeds(SEED)
        t_train = time.time()
        agent, ep_rewards, ep_losses = run_training(
            best_params, time_matrix, reward_matrix_penalized,
            noise_sigma, NUM_NODES,
        )
        train_time = time.time() - t_train
        print(f"Training time: {train_time:.1f} s")

        agent.save(os.path.join(cwd, f"agent_checkpoint_{NUM_NODES}nodes.pt"))

        # 4. Solver comparison
        results_df, timing = run_solver_comparison(
            agent, time_matrix, reward_matrix,
            reward_matrix_penalized, noise_sigma, NUM_NODES,
        )

        # 5. Print summary
        def avg_valid(col, valid_col):
            mask = results_df[valid_col]
            return results_df.loc[mask, col].mean() if mask.any() else float('nan')

        print(f"\n{'='*60}")
        print(f"  SUMMARY — {NUM_NODES} nodes")
        print(f"{'='*60}")
        print(f"  MIP avg reward    : {avg_valid('MIP Reward',       'MIP Valid'):.1f}")
        print(f"  DRL avg reward    : {avg_valid('DRL Reward',       'DRL Valid'):.1f}")
        print(f"  Greedy avg reward : {avg_valid('Heuristic Reward', 'Heuristic Valid'):.1f}")
        print(f"  2-Opt avg reward  : {avg_valid('2Opt Reward',      '2Opt Valid'):.1f}")
        print(f"  GA avg reward     : {avg_valid('LNS Reward',       'LNS Valid'):.1f}")
        print(f"  Training time     : {train_time:.1f} s")
        print(f"  DRL avg inference : {np.mean(timing['drl_times'])*1000:.1f} ms")
        print(f"  MIP avg inference : {np.mean(timing['mip_times'])*1000:.1f} ms")

        gap_data = results_df.loc[
            results_df['MIP Valid'] & results_df['DRL Valid'], 'DRL Gap (%)'
        ].dropna()
        if len(gap_data) > 0:
            print(f"  DRL avg gap vs MIP: {gap_data.mean():.2f}%")
            print(f"  DRL max gap vs MIP: {gap_data.max():.2f}%")

        summary_rows.append({
            'Node Size':          NUM_NODES,
            'MIP Avg Reward':     avg_valid('MIP Reward',       'MIP Valid'),
            'DRL Avg Reward':     avg_valid('DRL Reward',       'DRL Valid'),
            'Heuristic Avg':      avg_valid('Heuristic Reward', 'Heuristic Valid'),
            '2Opt Avg':           avg_valid('2Opt Reward',      '2Opt Valid'),
            'GA Avg':             avg_valid('LNS Reward',       'LNS Valid'),
            'DRL Training Time':  train_time,
            'DRL Avg Gap (%)':    gap_data.mean() if len(gap_data) > 0 else float('nan'),
        })

        # 6. Save outputs
        suffix      = f"_stochastic_sigma{NOISE_FRACTION:.0%}" if STOCHASTIC_MODE else "_deterministic"
        excel_path  = os.path.join(cwd, f"DRL_Routing_Summary{suffix}.xlsx")
        plot_path   = os.path.join(cwd, f"DRL_Training_Diagnostics{suffix}.png")

        save_results(results_df, summary_rows, excel_path)
        plot_diagnostics(ep_rewards, ep_losses, results_df, NUM_NODES, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
