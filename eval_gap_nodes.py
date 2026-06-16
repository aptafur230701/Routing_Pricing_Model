"""
eval_gap_nodes.py
=================
Re-evalúa solo los nodos con gap >20% usando beam_width elevado,
sin reentrenar nada. Sirve para diagnosticar si el gap es problema
de inferencia (beam_width muy pequeño) o de política aprendida.

Uso
---
    python eval_gap_nodes.py                    # beam_width=10, nodos por defecto
    python eval_gap_nodes.py --beam 15          # beam_width=15
    python eval_gap_nodes.py --nodes 0 6 7      # nodos específicos
    python eval_gap_nodes.py --beam 12 --nodes 0 6 7 12 15 18
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--beam",  type=int, default=10,
                    help="beam_width a usar (default: 10)")
parser.add_argument("--nodes", type=int, nargs="+",
                    default=[0, 6, 7, 12, 15, 18],
                    help="nodos a re-evaluar (default: 0 6 7 12 15 18)")
parser.add_argument("--num_nodes", type=int, default=20,
                    help="tamaño del modelo / problema (default: 20)")
parser.add_argument("--samples", type=int, default=5,
                    help="rollouts estocásticos por nodo (default: 5)")
args = parser.parse_args()

BEAM_WIDTH  = args.beam
GAP_NODES   = args.nodes
NUM_NODES   = args.num_nodes
N_SAMPLES   = args.samples

# ── Imports del proyecto ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, MAX_DURATION, SEED, TRAIN_DAYS
from am_agent import AMRoutingAgent
from problem_data import load_matrices
from Solvers import solve_label_setting_exact, simulate_route_reward

# ── Carga checkpoint ──────────────────────────────────────────────────────────
checkpoint_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
checkpoint_path = os.path.join(checkpoint_dir, f"am_checkpoint_{NUM_NODES}nodes.pt")

if not os.path.exists(checkpoint_path):
    sys.exit(f"ERROR: checkpoint no encontrado → {checkpoint_path}")

ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
agent = AMRoutingAgent(num_nodes=NUM_NODES, device=DEVICE)
agent.load_state_dict(ckpt["agent"])
agent.eval()
print(f"Checkpoint cargado: {checkpoint_path}")

# ── Carga datos ───────────────────────────────────────────────────────────────
time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
    ltr_stack, trucks_stack, avail_prob_arr, _ = load_matrices(NUM_NODES)

# Split eval (mismos días que usa main.py)
rate_eval  = rate_stack[TRAIN_DAYS:]
loads_eval = loads_stack[TRAIN_DAYS:]
num_days   = rate_eval.shape[0]

time_matrix_np = np.array(time_matrix, dtype=float)

# Reproducir los mismos day_idx que usa run_solver_comparison
rng         = np.random.default_rng(SEED)
all_indices = rng.integers(0, num_days, size=NUM_NODES)  # genera para todos los nodos
day_for     = {s: int(all_indices[s]) for s in range(NUM_NODES)}

# ── Eval ──────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Re-evaluación de nodos con gap >20%")
print(f"  Nodos    : {GAP_NODES}")
print(f"  beam_width actual (config): 5  →  usando: {BEAM_WIDTH}")
print(f"  Modelo   : {NUM_NODES} nodos  |  Muestras estocásticas: {N_SAMPLES}")
print(f"{'='*60}\n")

rows = []
for s in GAP_NODES:
    day_idx     = day_for[s]
    abs_day_idx = TRAIN_DAYS + day_idx
    print(f"Nodo {s}  |  eval day {day_idx} (abs {abs_day_idx})")

    # ── DRL Det (beam_width elevado, sin Bernoulli) ──────────────────────────
    agent.eval()
    det_route, _, _ = agent.beam_search_dynamic(
        s, day_idx,
        time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
        MAX_DURATION,
        beam_width=BEAM_WIDTH,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
        avail_prob_arr=None,   # determinista
    )
    agent.train()

    det_valid = det_route is not None
    if det_valid:
        det_reward, det_duration = simulate_route_reward(
            det_route, s, day_idx,
            time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
            avail_prob_arr=None,
        )
    else:
        det_reward, det_duration = -np.inf, np.inf

    # ── DRL Real (beam_width elevado, con Bernoulli) — mejor de N_SAMPLES ────
    best_route, best_reward, best_duration = None, -np.inf, np.inf
    for _ in range(N_SAMPLES):
        agent.eval()
        r, rew, dur = agent.beam_search_dynamic(
            s, day_idx,
            time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
            MAX_DURATION,
            beam_width=BEAM_WIDTH,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=avail_prob_arr,
        )
        agent.train()
        if r is not None and rew > best_reward:
            best_route, best_reward, best_duration = r, rew, dur

    # ── LS-Exact (referencia) ─────────────────────────────────────────────────
    t0 = time.time()
    ls_status, ls_route, ls_reward, ls_duration = solve_label_setting_exact(
        s, time_matrix_np, rate_eval, loads_eval,
        distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
        start_day_idx=day_idx,
        time_limit_seconds=300,
    )
    ls_time = time.time() - t0
    ls_valid = ls_status in ("Optimal", "Time-Limited") and ls_route is not None
    if not ls_valid:
        ls_reward = -np.inf

    # ── Gap ───────────────────────────────────────────────────────────────────
    gap_det  = (ls_reward - det_reward)  / abs(ls_reward) * 100 if ls_valid and det_valid  and ls_reward != 0 else float("nan")
    gap_real = (ls_reward - best_reward) / abs(ls_reward) * 100 if ls_valid and best_route is not None and ls_reward != 0 else float("nan")

    print(f"  DRL Det  (beam={BEAM_WIDTH}): {det_route}  reward={det_reward:.1f}  gap vs LS-Exact={gap_det:+.1f}%")
    print(f"  DRL Real (beam={BEAM_WIDTH}): {best_route}  reward={best_reward:.1f}  gap vs LS-Exact={gap_real:+.1f}%")
    print(f"  LS-Exact [{ls_status}]:       {ls_route}  reward={ls_reward:.1f}  ({ls_time:.1f}s)\n")

    rows.append({
        "node": s, "day_idx": day_idx,
        "beam": BEAM_WIDTH,
        "det_reward": det_reward, "det_route": det_route, "gap_det": gap_det,
        "real_reward": best_reward, "real_route": best_route, "gap_real": gap_real,
        "ls_reward": ls_reward, "ls_status": ls_status,
    })

# ── Resumen ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  RESUMEN  beam_width={BEAM_WIDTH}")
print(f"  {'Nodo':>5}  {'DRL Det':>10}  {'Gap Det%':>9}  {'DRL Real':>10}  {'Gap Real%':>10}  {'LS-Exact':>10}")
print(f"  {'-'*65}")
for r in rows:
    print(f"  {r['node']:>5}  {r['det_reward']:>10.1f}  {r['gap_det']:>+9.1f}  "
          f"{r['real_reward']:>10.1f}  {r['gap_real']:>+10.1f}  {r['ls_reward']:>10.1f}")
valid_det  = [r["gap_det"]  for r in rows if not np.isnan(r["gap_det"])]
valid_real = [r["gap_real"] for r in rows if not np.isnan(r["gap_real"])]
print(f"  {'AVG':>5}  {'':>10}  {np.mean(valid_det):>+9.1f}  {'':>10}  {np.mean(valid_real):>+10.1f}")
print(f"{'='*60}")
print("\nInterpretación:")
print("  gap << 20%  → beam_width era el problema (inferencia)")
print("  gap >> 20%  → el problema es la política aprendida")
