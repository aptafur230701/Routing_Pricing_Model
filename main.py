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
from am_agent import AMRoutingAgent


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
    reward_global_p95=1.0,
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
        reward_global_p95=reward_global_p95,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f} s")

    torch.save(
        {"agent": agent_am.state_dict(), "critic": critic.state_dict()},
        os.path.join(cwd, f"am_checkpoint_{NUM_NODES}nodes.pt"),
    )
    return agent_am, ep_rewards, ep_losses, train_time, training_log


def _load_agent(checkpoint_dir: str, num_nodes: int) -> "AMRoutingAgent":
    """Carga un agente entrenado desde checkpoint sin reentrenar."""
    path = os.path.join(checkpoint_dir, f"am_checkpoint_{num_nodes}nodes.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")
    ckpt  = torch.load(path, map_location=DEVICE)
    agent = AMRoutingAgent(num_nodes, device=DEVICE)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    print(f"Checkpoint cargado: {path}")
    return agent


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

    def _gap_series(gap_col, solver_valid_col):
        return results_df.loc[
            results_df["MIP-Exact Valid"] & results_df[solver_valid_col], gap_col
        ].dropna()

    drl_gap     = _gap_series("MIP-Exact Gap vs DRL Real (%)",  "DRL Real Valid")
    hga_gap     = _gap_series("MIP-Exact Gap vs HGA-LNS (%)",   "HGA-LNS Valid")
    rh_gap      = _gap_series("MIP-Exact Gap vs RH-Greedy (%)", "RH-Greedy Valid")

    def _gap_stats(series):
        if len(series) > 0:
            return series.mean(), series.max()
        return float("nan"), float("nan")

    drl_avg_gap, drl_max_gap = _gap_stats(drl_gap)
    hga_avg_gap, hga_max_gap = _gap_stats(hga_gap)
    rh_avg_gap,  rh_max_gap  = _gap_stats(rh_gap)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {NUM_NODES} nodes  [{label}]")
    print(f"{'='*60}")
    print(f"  MIP-Exact avg reward   : {avg_valid('MIP-Exact Reward', 'MIP-Exact Valid'):.1f}")
    print(f"  DRL Real avg reward    : {avg_valid('DRL Real Reward',  'DRL Real Valid'):.1f}")
    print(f"  HGA-LNS avg reward     : {avg_valid('HGA-LNS Reward',   'HGA-LNS Valid'):.1f}")
    print(f"  RH-Greedy avg reward   : {avg_valid('RH-Greedy Reward', 'RH-Greedy Valid'):.1f}")
    print(f"  Training time          : {train_time:.1f} s")
    print(f"  DRL Real avg inference : {np.mean(timing['drl_real_times'])*1000:.1f} ms")
    print(f"  HGA-LNS avg inference  : {np.mean(timing['hga_lns_times'])*1000:.1f} ms")
    print(f"  RH-Greedy avg inference: {np.mean(timing['rh_greedy_times'])*1000:.1f} ms")
    print(f"  MIP-Exact avg inference: {np.mean(timing['mip_exact_times'])*1000:.1f} ms")
    print(f"  --- Gaps vs MIP-Exact ---")
    print(f"  DRL Real  : avg {drl_avg_gap:.2f}%  max {drl_max_gap:.2f}%")
    print(f"  HGA-LNS   : avg {hga_avg_gap:.2f}%  max {hga_max_gap:.2f}%")
    print(f"  RH-Greedy : avg {rh_avg_gap:.2f}%  max {rh_max_gap:.2f}%")

    summary_rows.append({
        "Model":                           label,
        "Node Size":                       NUM_NODES,
        "MIP-Exact Avg Reward":            avg_valid("MIP-Exact Reward", "MIP-Exact Valid"),
        "DRL Real Avg Reward":             avg_valid("DRL Real Reward",  "DRL Real Valid"),
        "HGA-LNS Avg Reward":              avg_valid("HGA-LNS Reward",   "HGA-LNS Valid"),
        "RH-Greedy Avg Reward":            avg_valid("RH-Greedy Reward", "RH-Greedy Valid"),
        "DRL Training Time (s)":           train_time,
        "DRL Real Avg Inference (ms)":     np.mean(timing["drl_real_times"]) * 1000,
        "HGA-LNS Avg Inference (ms)":      np.mean(timing["hga_lns_times"])  * 1000,
        "RH-Greedy Avg Inference (ms)":    np.mean(timing["rh_greedy_times"]) * 1000,
        "MIP-Exact Avg Inference (ms)":        np.mean(timing["mip_exact_times"]) * 1000,
        "DRL Real Avg Gap vs MIP-Exact (%)":   drl_avg_gap,
        "DRL Real Max Gap vs MIP-Exact (%)":   drl_max_gap,
        "HGA-LNS Avg Gap vs MIP-Exact (%)":    hga_avg_gap,
        "HGA-LNS Max Gap vs MIP-Exact (%)":    hga_max_gap,
        "RH-Greedy Avg Gap vs MIP-Exact (%)":  rh_avg_gap,
        "RH-Greedy Max Gap vs MIP-Exact (%)":  rh_max_gap,
    })

    excel_path = os.path.join(cwd, f"DRL_Routing_Summary_{label}.xlsx")
    plot_path  = os.path.join(cwd, f"DRL_Training_Diagnostics_{label}.png")

    save_results(results_df, summary_rows, excel_path)
    if ep_rewards:
        plot_diagnostics(ep_rewards, ep_losses, results_df, NUM_NODES, plot_path)

    return results_df


def main():
    # ── Cambia estos valores según lo que quieras hacer ───────────────────────
    NUM_NODES  = 10     # opciones: 10 · 20 · 35 · 50 · 75 · 97
    EVAL_ONLY  = True  # True: carga checkpoint y salta entrenamiento
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
        ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95 = load_matrices(NUM_NODES)

    # ── Train / eval split (temporal — el modelo no ve los días de eval) ──────
    rate_train  = rate_stack[:TRAIN_DAYS]
    loads_train = loads_stack[:TRAIN_DAYS]
    rate_eval   = rate_stack[TRAIN_DAYS:]
    loads_eval  = loads_stack[TRAIN_DAYS:]

    if EVAL_ONLY:
        agent_am  = _load_agent(checkpoint_dir, NUM_NODES)
        ep_r, ep_l, t, training_log = [], [], 0.0, []
    else:
        agent_pretrained, critic_pretrained = build_agent_for_training(checkpoint_dir, NUM_NODES)
        agent_am, ep_r, ep_l, t, training_log = _run_am(
            NUM_NODES, time_matrix, rate_train, loads_train,
            distance_arr, diesel_arr, checkpoint_dir,
            agent_pretrained, critic_pretrained,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=avail_prob_arr,
            reward_global_p95=reward_global_p95,
        )
        ppo_plot_path = os.path.join(output_dir, "PPO_Diagnostics_AM.png")
        plot_ppo_diagnostics(training_log, NUM_NODES, ppo_plot_path)

    _evaluate_and_report(
        agent_am, NUM_NODES, time_matrix, rate_eval, loads_eval,
        distance_arr, diesel_arr, ep_r, ep_l, t,
        summary_rows, output_dir, label="AM",
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
    )

    print(f"\nDone! Checkpoint guardado en: checkpoints/am_checkpoint_{NUM_NODES}nodes.pt")


if __name__ == "__main__":
    main()
