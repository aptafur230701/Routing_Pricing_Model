"""
main.py
=======
Entry point for the AM-PPO pipeline.

  1. Load data          (problem_data.py)
  2. Training           (am_training.py)
  3. Evaluation         (evaluation.py)
  4. Save results       (evaluation.py)

Uso
---
  Cambia NUM_NODES abajo según la rama en la que estés:
    train-10nodes  →  NUM_NODES = 10
    train-20nodes  →  NUM_NODES = 20
    train-35nodes  →  NUM_NODES = 35
    train-50nodes  →  NUM_NODES = 50
    train-75nodes  →  NUM_NODES = 75
    train-97nodes  →  NUM_NODES = 97

  Luego ejecuta simplemente:
    python main.py

Checkpoints (transfer learning)
--------------------------------
  Los .pt se guardan en  checkpoints/  (raíz del proyecto, ignorada por git).
  Para usar transfer learning, copia el .pt del tamaño anterior en esa carpeta
  antes de entrenar (o entrena en orden y se guarda sólo).
"""

import os
import random
import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

import sys
print(sys.executable)

from config import (
    SEED, DEVICE,
    get_episodes_per_node, TRAIN_DAYS,
)
from problem_data import load_matrices
from evaluation import (
    run_solver_comparison, save_results, plot_diagnostics, plot_ppo_diagnostics,
)
from transfer_learning import build_agent_for_training


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _run_am(
    NUM_NODES, time_matrix, rate_stack, loads_stack,
    distance_arr, diesel_arr, cwd,
    pretrained_agent=None, pretrained_critic=None,
    ltr_stack=None, trucks_stack=None, avail_prob_arr=None,
):
    """Pipeline AM: entrenamiento PPO → checkpoint."""
    from am_training import run_am_training

    episodes_per_node = get_episodes_per_node(NUM_NODES)
    num_episodes      = episodes_per_node * NUM_NODES

    print(f"Device           : {DEVICE}")
    print(f"Training episodes: {num_episodes}")

    set_seeds(SEED)
    t0 = time.time()
    agent_am, critic, ep_rewards, ep_losses, training_log = run_am_training(
        time_matrix, rate_stack, loads_stack,
        distance_arr, diesel_arr, NUM_NODES,
        pretrained_agent=pretrained_agent,
        pretrained_critic=pretrained_critic,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    torch.save(
        {"agent": agent_am.state_dict(), "critic": critic.state_dict()},
        os.path.join(cwd, f"am_checkpoint_{NUM_NODES}nodes.pt"),
    )
    return agent_am, ep_rewards, ep_losses, train_time, training_log


def _evaluate_and_report(
    agent, NUM_NODES, time_matrix, rate_stack, loads_stack,
    distance_arr, diesel_arr, ep_rewards, ep_losses,
    train_time, summary_rows, cwd, label,
    ltr_stack=None, trucks_stack=None, avail_prob_arr=None,
):
    """Comparación de solvers + resumen + guardado de archivos."""
    results_df, timing = run_solver_comparison(
        agent, time_matrix,
        rate_stack, loads_stack, distance_arr, diesel_arr,
        NUM_NODES,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
    )

    def avg_valid(col, valid_col):
        mask = results_df[valid_col]
        return results_df.loc[mask, col].mean() if mask.any() else float("nan")

    gap_data = results_df.loc[
        results_df["MIP Valid"] & results_df["DRL Valid"], "DRL Gap (%)"
    ].dropna()

    drl_real_gap_oracle = results_df.loc[
        results_df["Oracle Valid"] & results_df["DRL Real Valid"],
        "Oracle Gap vs DRL Real (%)"
    ].dropna()

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {NUM_NODES} nodes  [{label}]")
    print(f"{'='*60}")
    print(f"  MIP-Oracle avg reward  : {avg_valid('Oracle Reward',    'Oracle Valid'):.1f}")
    print(f"  MIP avg reward         : {avg_valid('MIP Reward',       'MIP Valid'):.1f}")
    print(f"  DRL Real avg reward    : {avg_valid('DRL Real Reward',  'DRL Real Valid'):.1f}")
    print(f"  DRL Det avg reward     : {avg_valid('DRL Det Reward',   'DRL Valid'):.1f}")
    print(f"  2-Opt avg reward       : {avg_valid('2Opt Reward',      '2Opt Valid'):.1f}")
    print(f"  GA avg reward          : {avg_valid('GA Reward',        'GA Valid'):.1f}")
    print(f"  LNS avg reward         : {avg_valid('LNS Reward',       'LNS Valid'):.1f}")
    print(f"  HGA-LNS avg reward     : {avg_valid('HGA-LNS Reward',   'HGA-LNS Valid'):.1f}")
    print(f"  RH-Greedy avg reward   : {avg_valid('RH-Greedy Reward', 'RH-Greedy Valid'):.1f}")
    print(f"  Training time          : {train_time:.1f} s")
    print(f"  DRL avg inference      : {np.mean(timing['drl_times'])*1000:.1f} ms")
    print(f"  MIP avg inference      : {np.mean(timing['mip_times'])*1000:.1f} ms")
    if len(gap_data) > 0:
        print(f"  DRL Det avg gap vs MIP : {gap_data.mean():.2f}%")
        print(f"  DRL Det max gap vs MIP : {gap_data.max():.2f}%")
    if len(drl_real_gap_oracle) > 0:
        print(f"  DRL Real avg gap vs Oracle: {drl_real_gap_oracle.mean():.2f}%")
        print(f"  DRL Real max gap vs Oracle: {drl_real_gap_oracle.max():.2f}%")

    summary_rows.append({
        "Model":                          label,
        "Node Size":                      NUM_NODES,
        "Oracle Avg Reward":              avg_valid("Oracle Reward",    "Oracle Valid"),
        "MIP Avg Reward":                 avg_valid("MIP Reward",       "MIP Valid"),
        "DRL Real Avg Reward":            avg_valid("DRL Real Reward",  "DRL Real Valid"),
        "DRL Det Avg Reward":             avg_valid("DRL Det Reward",   "DRL Valid"),
        "2Opt Avg":                       avg_valid("2Opt Reward",      "2Opt Valid"),
        "GA Avg":                         avg_valid("GA Reward",        "GA Valid"),
        "LNS Avg":                        avg_valid("LNS Reward",       "LNS Valid"),
        "HGA-LNS Avg":                    avg_valid("HGA-LNS Reward",  "HGA-LNS Valid"),
        "RH-Greedy Avg":                  avg_valid("RH-Greedy Reward", "RH-Greedy Valid"),
        "DRL Training Time":              train_time,
        "DRL Det Avg Gap vs MIP (%)":     gap_data.mean() if len(gap_data) > 0 else float("nan"),
        "DRL Real Avg Gap vs Oracle (%)": drl_real_gap_oracle.mean() if len(drl_real_gap_oracle) > 0 else float("nan"),
        "DRL Real Max Gap vs Oracle (%)": drl_real_gap_oracle.max()  if len(drl_real_gap_oracle) > 0 else float("nan"),
    })

    excel_path = os.path.join(cwd, f"DRL_Routing_Summary_{label}.xlsx")
    plot_path  = os.path.join(cwd, f"DRL_Training_Diagnostics_{label}.png")

    save_results(results_df, summary_rows, excel_path)
    plot_diagnostics(ep_rewards, ep_losses, results_df, NUM_NODES, plot_path)

    return results_df


def main():
    # ── Cambia este valor según la rama en la que estés ────────────────────────
    NUM_NODES = 10   # opciones: 10 · 20 · 35 · 50 · 75 · 97
    # ──────────────────────────────────────────────────────────────────────────

    cwd          = os.path.dirname(os.path.abspath(__file__))
    summary_rows = []

    print(f"\n{'='*60}")
    print(f"  AM — {NUM_NODES} nodes")
    print(f"{'='*60}")

    set_seeds(SEED)

    output_dir = os.path.join(cwd, f"results_{NUM_NODES}nodes")
    os.makedirs(output_dir, exist_ok=True)

    # Carpeta compartida de checkpoints (para transfer learning entre ramas)
    checkpoint_dir = os.path.join(cwd, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_prob_arr = load_matrices(NUM_NODES)

    # ── Train / eval split (temporal — el modelo no ve los días de eval) ──────
    rate_train  = rate_stack[:TRAIN_DAYS]
    loads_train = loads_stack[:TRAIN_DAYS]
    rate_eval   = rate_stack[TRAIN_DAYS:]
    loads_eval  = loads_stack[TRAIN_DAYS:]

    agent_pretrained, critic_pretrained = build_agent_for_training(checkpoint_dir, NUM_NODES)
    agent_am, ep_r, ep_l, t, training_log = _run_am(
        NUM_NODES, time_matrix, rate_train, loads_train,
        distance_arr, diesel_arr, checkpoint_dir,
        agent_pretrained, critic_pretrained,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
    )
    _evaluate_and_report(
        agent_am, NUM_NODES, time_matrix, rate_eval, loads_eval,
        distance_arr, diesel_arr, ep_r, ep_l, t,
        summary_rows, output_dir, label="AM",
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
    )
    ppo_plot_path = os.path.join(output_dir, "PPO_Diagnostics_AM.png")
    plot_ppo_diagnostics(training_log, NUM_NODES, ppo_plot_path)

    print(f"\nDone! Checkpoint guardado en: checkpoints/am_checkpoint_{NUM_NODES}nodes.pt")


if __name__ == "__main__":
    main()
