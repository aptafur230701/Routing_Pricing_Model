"""
main_ddqn.py
============
Self-contained entry point for the DDQN pipeline.

Usage
-----
  python ddqn_pipeline/main_ddqn.py
"""

import os
import sys
import random
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Path setup: ensure ddqn_pipeline/ is importable regardless of cwd ─────────
_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

print(sys.executable)

from config import (
    SEED, DEVICE, STOCHASTIC_MODE, NOISE_FRACTION,
    REWARD_SCALE_FACTOR,
    get_episodes_per_node,
)
from state import get_state_size
from problem_data import load_matrices
from evaluation import run_solver_comparison, save_results, plot_diagnostics
from tuning import run_optuna
from training import run_training


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _run_ddqn(
    NUM_NODES, time_matrix, rate_stack, loads_stack,
    distance_arr, diesel_arr, noise_sigma, cwd,
):
    """Pipeline completo DDQN: Optuna → entrenamiento → checkpoint."""
    state_size          = get_state_size(NUM_NODES)
    episodes_per_node   = get_episodes_per_node(NUM_NODES)
    num_episodes        = episodes_per_node * NUM_NODES
    epsilon_decay_steps = num_episodes * NUM_NODES

    print(f"Device           : {DEVICE}")
    print(f"State size       : {state_size}  (2 + {NUM_NODES} visited + 2 step)")
    print(f"Training episodes: {num_episodes}")

    print("\n--- Optuna (30 trials) ---")
    best_params = run_optuna(
        time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        noise_sigma, NUM_NODES, epsilon_decay_steps, n_trials=30,
    )

    set_seeds(SEED)
    t0 = time.time()
    agent, ep_rewards, ep_losses = run_training(
        best_params, time_matrix, rate_stack, loads_stack,
        distance_arr, diesel_arr, noise_sigma, NUM_NODES,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    agent.save(os.path.join(cwd, f"ddqn_checkpoint_{NUM_NODES}nodes.pt"))
    return agent, ep_rewards, ep_losses, train_time


def _evaluate_and_report(
    agent, NUM_NODES, time_matrix, rate_stack, loads_stack,
    distance_arr, diesel_arr, noise_sigma, ep_rewards, ep_losses,
    train_time, summary_rows, cwd, label,
):
    """Comparación de solvers + resumen + guardado de archivos."""
    results_df, timing = run_solver_comparison(
        agent, time_matrix,
        rate_stack, loads_stack, distance_arr, diesel_arr,
        noise_sigma, NUM_NODES,
    )

    def avg_valid(col, valid_col):
        mask = results_df[valid_col]
        return results_df.loc[mask, col].mean() if mask.any() else float("nan")

    gap_data = results_df.loc[
        results_df["MIP Valid"] & results_df["DRL Valid"], "DRL Gap (%)"
    ].dropna()

    stoch_gap_data = results_df.loc[
        results_df["MIP Valid"] & (results_df["DRL Stoch Valid%"] > 0), "DRL Stoch Gap (%)"
    ].dropna()

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {NUM_NODES} nodes  [{label}]")
    print(f"{'='*60}")
    print(f"  MIP avg reward    : {avg_valid('MIP Reward',       'MIP Valid'):.1f}")
    print(f"  DRL avg reward    : {avg_valid('DRL Det Reward',    'DRL Valid'):.1f}")
    print(f"  Greedy avg reward : {avg_valid('Heuristic Reward', 'Heuristic Valid'):.1f}")
    print(f"  2-Opt avg reward  : {avg_valid('2Opt Reward',      '2Opt Valid'):.1f}")
    print(f"  GA avg reward     : {avg_valid('GA Reward',          'GA Valid'):.1f}")
    print(f"  LNS avg reward    : {avg_valid('LNS Reward',         'LNS Valid'):.1f}")
    print(f"  HGA-LNS avg reward: {avg_valid('HGA-LNS Reward',     'HGA-LNS Valid'):.1f}")
    print(f"  Training time     : {train_time:.1f} s")
    print(f"  DRL avg inference : {np.mean(timing['drl_times'])*1000:.1f} ms")
    print(f"  MIP avg inference : {np.mean(timing['mip_times'])*1000:.1f} ms")
    if len(gap_data) > 0:
        print(f"  DRL avg gap vs MIP: {gap_data.mean():.2f}%")
        print(f"  DRL max gap vs MIP: {gap_data.max():.2f}%")
    print(f"  DRL avg stoch gap vs MIP: {stoch_gap_data.mean():.2f}%" if len(stoch_gap_data) > 0 else "  DRL avg stoch gap vs MIP: N/A")

    summary_rows.append({
        "Model":                  label,
        "Node Size":              NUM_NODES,
        "MIP Avg Reward":         avg_valid("MIP Reward",       "MIP Valid"),
        "DRL Avg Reward":         avg_valid("DRL Det Reward",   "DRL Valid"),
        "Heuristic Avg":          avg_valid("Heuristic Reward", "Heuristic Valid"),
        "2Opt Avg":               avg_valid("2Opt Reward",      "2Opt Valid"),
        "GA Avg":                 avg_valid("GA Reward",        "GA Valid"),
        "LNS Avg":                avg_valid("LNS Reward",       "LNS Valid"),
        "HGA-LNS Avg":            avg_valid("HGA-LNS Reward",  "HGA-LNS Valid"),
        "DRL Training Time":      train_time,
        "DRL Avg Gap (%)":        gap_data.mean() if len(gap_data) > 0 else float("nan"),
        "DRL Avg Stoch Gap (%)":  stoch_gap_data.mean() if len(stoch_gap_data) > 0 else float("nan"),
    })

    suffix     = f"_stochastic_sigma{NOISE_FRACTION:.0%}" if STOCHASTIC_MODE else "_deterministic"
    excel_path = os.path.join(cwd, f"DRL_Routing_Summary{suffix}_{label}.xlsx")
    plot_path  = os.path.join(cwd, f"DRL_Training_Diagnostics{suffix}_{label}.png")

    save_results(results_df, summary_rows, excel_path)
    plot_diagnostics(ep_rewards, ep_losses, results_df, NUM_NODES, plot_path)

    return results_df


def main():
    cwd          = os.path.dirname(os.path.abspath(__file__))
    summary_rows = []

    for NUM_NODES in [10]:
        print(f"\n{'='*60}")
        print(f"  DDQN — {NUM_NODES} nodes")
        print(f"{'='*60}")

        set_seeds(SEED)

        time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, noise_sigma = \
            load_matrices(NUM_NODES)

        agent_ddqn, ep_r, ep_l, t = _run_ddqn(
            NUM_NODES, time_matrix, rate_stack, loads_stack,
            distance_arr, diesel_arr, noise_sigma, cwd,
        )
        _evaluate_and_report(
            agent_ddqn, NUM_NODES, time_matrix, rate_stack, loads_stack,
            distance_arr, diesel_arr, noise_sigma, ep_r, ep_l, t,
            summary_rows, cwd, label="DDQN",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
