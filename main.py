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
    train-100nodes  →  NUM_NODES = 100

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
from stats_analysis import append_stats_sheets_to_excel
from transfer_learning import build_agent_for_training
from am_agent import AMRoutingAgent
from convergence_check import check_convergence, print_convergence_report, save_convergence_to_excel


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
    n_days_per_node=1,
):
    """Comparación de solvers + resumen + guardado de archivos."""
    results_df, timing = run_solver_comparison(
        agent, time_matrix,
        rate_stack, loads_stack, distance_arr, diesel_arr,
        NUM_NODES,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
        n_days_per_node=n_days_per_node,
    )

    def avg_valid(col, valid_col):
        mask = results_df[valid_col]
        return results_df.loc[mask, col].mean() if mask.any() else float("nan")

    def _gap_stats(series):
        if len(series) > 0:
            return series.mean(), series.max()
        return float("nan"), float("nan")

    both_valid = results_df["DRL Det Valid"] & results_df["DRL Real Valid"]
    real_vs_det_series = results_df.loc[both_valid].apply(
        lambda r: (r["DRL Det Reward"] - r["DRL Real Reward"]) / abs(r["DRL Det Reward"]) * 100
        if r["DRL Det Reward"] != 0 else float("nan"), axis=1
    ).dropna()
    real_vs_det_avg, _ = _gap_stats(real_vs_det_series)

    drl_vs_rh_stoch_valid = results_df["DRL Real Valid"] & results_df["RH-Greedy Real Valid"]
    _m_drl_rh  = results_df.loc[drl_vs_rh_stoch_valid, "DRL Real Reward"].mean()
    _m_rh      = results_df.loc[drl_vs_rh_stoch_valid, "RH-Greedy Real Reward"].mean()
    drl_vs_rh_stoch_avg = (_m_drl_rh - _m_rh) / abs(_m_rh) * 100 if _m_rh != 0 else float("nan")

    drl_vs_rhlr_valid = results_df["DRL Real Valid"] & results_df["RH-Lookahead Real Valid"]
    _m_drl_rhlr = results_df.loc[drl_vs_rhlr_valid, "DRL Real Reward"].mean()
    _m_rhlr     = results_df.loc[drl_vs_rhlr_valid, "RH-Lookahead Real Reward"].mean()
    drl_vs_rhlr_avg = (_m_drl_rhlr - _m_rhlr) / abs(_m_rhlr) * 100 if _m_rhlr != 0 else float("nan")

    drl_vs_mc_valid = results_df["DRL Real Valid"] & results_df["MC-Rollout Valid"]
    _m_drl_mc = results_df.loc[drl_vs_mc_valid, "DRL Real Reward"].mean()
    _m_mc     = results_df.loc[drl_vs_mc_valid, "MC-Rollout Reward"].mean()
    drl_vs_mc_avg = (_m_drl_mc - _m_mc) / abs(_m_mc) * 100 if _m_mc != 0 else float("nan")

    drl_both_ms = np.mean(timing["drl_det_times"] + timing["drl_real_times"]) * 1000

    ls_drl_det_valid = results_df["LS-Exact Valid"] & results_df["DRL Det Valid"]
    drl_det_gap_series = results_df.loc[ls_drl_det_valid].apply(
        lambda r: (r["LS-Exact Reward"] - r["DRL Det Reward"]) / abs(r["LS-Exact Reward"]) * 100
        if r["LS-Exact Reward"] != 0 else float("nan"), axis=1
    ).dropna()
    drl_det_avg_gap, _ = _gap_stats(drl_det_gap_series)

    ls_oracle_drl_real_valid = results_df["LS-Oracle Valid"] & results_df["DRL Real Valid"]
    drl_real_gap_series = results_df.loc[ls_oracle_drl_real_valid].apply(
        lambda r: (r["LS-Oracle Reward"] - r["DRL Real Reward"]) / abs(r["LS-Oracle Reward"]) * 100
        if r["LS-Oracle Reward"] != 0 else float("nan"), axis=1
    ).dropna()
    drl_real_avg_gap, _ = _gap_stats(drl_real_gap_series)

    print(f"\n{'='*60}")
    print(f"  SUMMARY — {NUM_NODES} nodes  [{label}]")
    print(f"{'='*60}")
    print(f"\n  Bloque 1 — Mundo determinista (comparación principal)")
    print(f"  LS-Exact avg reward    : {avg_valid('LS-Exact Reward', 'LS-Exact Valid'):>10.1f}")
    print(f"  DRL Det avg reward     : {avg_valid('DRL Det Reward', 'DRL Det Valid'):>10.1f}  | gap vs LS-Exact: avg ~{drl_det_avg_gap:.1f}%")
    print(f"  HGA-LNS avg reward     : {avg_valid('HGA-LNS Reward', 'HGA-LNS Valid'):>10.1f}")
    print(f"  RH-Greedy avg reward   : {avg_valid('RH-Greedy Reward', 'RH-Greedy Valid'):>10.1f}")
    print(f"  RH-Lookahead avg reward: {avg_valid('RH-Lookahead Reward', 'RH-Lookahead Valid'):>10.1f}")
    print(f"\n  Bloque 2 — Costo de ejecución estocástica")
    print(f"  LS-Oracle avg reward (cota clarividente): {avg_valid('LS-Oracle Reward', 'LS-Oracle Valid'):>10.1f}")
    print(f"  DRL Real avg reward      : {avg_valid('DRL Real Reward', 'DRL Real Valid'):>10.1f}  | gap vs DRL Det: avg ~{real_vs_det_avg:.1f}%  | gap vs LS-Oracle: avg ~{drl_real_avg_gap:.1f}%")
    print(f"  RH-Greedy Real avg reward     : {avg_valid('RH-Greedy Real Reward', 'RH-Greedy Real Valid'):>10.1f}")
    print(f"  RH-Lookahead Real avg reward  : {avg_valid('RH-Lookahead Real Reward', 'RH-Lookahead Real Valid'):>10.1f}")
    print(f"  MC-Rollout avg reward        : {avg_valid('MC-Rollout Reward', 'MC-Rollout Valid'):>10.1f}")
    print(f"  DRL Real advantage vs RH-Greedy Real:   avg ~{drl_vs_rh_stoch_avg:.1f}%")
    print(f"  DRL Real advantage vs RH-Lookahead Real: avg ~{drl_vs_rhlr_avg:.1f}%")
    print(f"  DRL Real advantage vs MC-Rollout:        avg ~{drl_vs_mc_avg:.1f}%")
    print(f"\n  Bloque 3 — Tiempo de inferencia")
    print(f"  DRL (Det+Real) : {drl_both_ms:>6.0f} ms")
    print(f"  HGA-LNS        : {np.mean(timing['hga_lns_times'])*1000:>6.0f} ms")
    print(f"  LS-Exact       : {np.mean(timing['ls_exact_times'])*1000:>6.0f} ms")
    print(f"  RH-Greedy      : {np.mean(timing['rh_greedy_times'])*1000:>6.0f} ms")
    print(f"  RH-Lookahead   : {np.mean(timing['rh_lookahead_times'])*1000:>6.0f} ms")
    print(f"  RH-Greedy Real : {np.mean(timing['rh_stoch_times'])*1000:>6.0f} ms")
    print(f"  RH-Lookahead Real: {np.mean(timing['rh_lookahead_stoch_times'])*1000:>6.0f} ms")
    print(f"  MC-Rollout       : {np.mean(timing['mc_rollout_times'])*1000:>6.0f} ms")
    print(f"  LS-Oracle      : {np.mean(timing['ls_oracle_times'])*1000:>6.0f} ms")
    print(f"  Training time  : {train_time:.1f} s")
    print(f"{'='*60}")


    summary_rows.append({
        "Model":                           label,
        "Node Size":                       NUM_NODES,
        "LS-Exact Avg Reward":             avg_valid("LS-Exact Reward", "LS-Exact Valid"),
        "DRL Det Avg Reward":              avg_valid("DRL Det Reward", "DRL Det Valid"),
        "DRL Det Gap vs LS-Exact (%)":     drl_det_avg_gap,
        "DRL Real Avg Reward":             avg_valid("DRL Real Reward",  "DRL Real Valid"),
        "LS-Oracle Avg Reward":            avg_valid("LS-Oracle Reward", "LS-Oracle Valid"),
        "DRL Real Gap vs LS-Oracle (%)":   drl_real_avg_gap,
        "LS-Oracle Avg Inference (ms)":    np.mean(timing["ls_oracle_times"]) * 1000,
        "HGA-LNS Avg Reward":              avg_valid("HGA-LNS Reward",   "HGA-LNS Valid"),
        "RH-Greedy Avg Reward":            avg_valid("RH-Greedy Reward",    "RH-Greedy Valid"),
        "RH-Lookahead Avg Reward":         avg_valid("RH-Lookahead Reward", "RH-Lookahead Valid"),
        "RH-Greedy Real Avg Reward":                    avg_valid("RH-Greedy Real Reward",    "RH-Greedy Real Valid"),
        "RH-Lookahead Real Avg Reward":                 avg_valid("RH-Lookahead Real Reward", "RH-Lookahead Real Valid"),
        "DRL Real Advantage vs RH-Greedy Real (%)":     drl_vs_rh_stoch_avg,
        "DRL Real Advantage vs RH-Lookahead Real (%)":  drl_vs_rhlr_avg,
        "DRL Training Time (s)":           train_time,
        "DRL Real Avg Inference (ms)":     np.mean(timing["drl_real_times"]) * 1000,
        "HGA-LNS Avg Inference (ms)":      np.mean(timing["hga_lns_times"])  * 1000,
        "LS-Exact Avg Inference (ms)":     np.mean(timing["ls_exact_times"]) * 1000,
        "RH-Greedy Avg Inference (ms)":    np.mean(timing["rh_greedy_times"]) * 1000,
        "RH-Lookahead Avg Inference (ms)":              np.mean(timing["rh_lookahead_times"]) * 1000,
        "RH-Lookahead Real Avg Inference (ms)":         np.mean(timing["rh_lookahead_stoch_times"]) * 1000,
        "MC-Rollout Avg Reward":                         avg_valid("MC-Rollout Reward", "MC-Rollout Valid"),
        "MC-Rollout Avg Inference (ms)":                 np.mean(timing["mc_rollout_times"]) * 1000,
        "DRL Real Advantage vs MC-Rollout (%)":          drl_vs_mc_avg,
    })

    excel_path = os.path.join(cwd, f"DRL_Routing_Summary_{label}.xlsx")
    plot_path  = os.path.join(cwd, f"DRL_Training_Diagnostics_{label}.png")

    save_results(results_df, summary_rows, excel_path)
    append_stats_sheets_to_excel(excel_path, results_df)
    if ep_rewards:
        plot_diagnostics(ep_rewards, ep_losses, results_df, NUM_NODES, plot_path)

    return results_df


def main():
    # ── Cambia estos valores según lo que quieras hacer ───────────────────────
    NUM_NODES       = 35     # opciones: 10 · 20 · 35 · 50 · 75 · 100
    EVAL_ONLY       = False   # True: carga checkpoint y salta entrenamiento
    N_DAYS_PER_NODE = 3      # días de evaluación por nodo (1 = comportamiento original)
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

    conv_result = None

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

        import pickle
        tlog_path = os.path.join(output_dir, f"training_log_{NUM_NODES}nodes.pkl")
        with open(tlog_path, "wb") as _f:
            pickle.dump(training_log, _f)
        print(f"training_log guardado en: {tlog_path}")

        if training_log:
            conv_result = check_convergence(training_log)
            print_convergence_report(conv_result)

    _evaluate_and_report(
        agent_am, NUM_NODES, time_matrix, rate_eval, loads_eval,
        distance_arr, diesel_arr, ep_r, ep_l, t,
        summary_rows, output_dir, label="AM",
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
        n_days_per_node=N_DAYS_PER_NODE,
    )

    if conv_result is not None:
        excel_path = os.path.join(output_dir, "DRL_Routing_Summary_AM.xlsx")
        save_convergence_to_excel(conv_result, NUM_NODES, excel_path)

    print(f"\nDone! Checkpoint guardado en: checkpoints/am_checkpoint_{NUM_NODES}nodes.pt")


if __name__ == "__main__":
    main()
