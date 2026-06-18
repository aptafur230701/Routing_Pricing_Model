"""
sweep_critic_value_beam.py
===========================
Barrido amplio: pruning intermedio del beam guiado por el critic
(score = total_reward + value_coef * V(s') * REWARD_SCALE_FACTOR) vs el
baseline sin critic (sort puro por total_reward), sobre el checkpoint ya
entrenado de 35 nodos. No reentrena nada.

Replica el esquema de evaluation.py::run_solver_comparison para los pares
(start_node, day_idx):
  - rng = np.random.default_rng(SEED)
  - day_indices[s, rep] = rng.choice(num_days_eval, size=n_days_per_node,
    replace=False), generado antes del loop principal.
  - day_idx es relativo al slice de evaluación (rate_stack[TRAIN_DAYS:]).

Compara DRL Det (avail_prob_arr=None, mundo determinista) con
beam_width = get_beam_width(NUM_NODES) — el mismo beam que usa run_solver_comparison
para DRL Det — entre critic=None y cada value_coef del barrido. El reward final
se recalcula con Solvers.simulate_route_reward para paridad bit-exacta, igual
que hace evaluation.py para DRL Det.
"""

import os
import numpy as np
import pandas as pd
import torch

from config import (
    SEED, DEVICE, TRAIN_DAYS, MAX_DURATION,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
    get_beam_width,
)
from problem_data import load_matrices
from am_agent import AMRoutingAgent
from critic_head import CriticHead
from Solvers import simulate_route_reward

NUM_NODES       = 35
N_DAYS_PER_NODE = 3
VALUE_COEFS     = [0.25, 0.5, 1.0]
OUTPUT_CSV      = "sweep_critic_value_beam.csv"


def load_agent_and_critic(checkpoint_dir, num_nodes):
    path = os.path.join(checkpoint_dir, f"am_checkpoint_{num_nodes}nodes.pt")
    ckpt = torch.load(path, map_location=DEVICE)
    agent = AMRoutingAgent(num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    critic = CriticHead(AM_D_H).to(DEVICE)
    critic.load_state_dict(ckpt["critic"])
    critic.eval()
    return agent, critic


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cwd = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(cwd, "checkpoints")
    agent, critic = load_agent_and_critic(checkpoint_dir, NUM_NODES)

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_prob_arr, _ = load_matrices(NUM_NODES)

    rate_eval  = rate_stack[TRAIN_DAYS:]
    loads_eval = loads_stack[TRAIN_DAYS:]
    num_days_eval = rate_eval.shape[0]
    time_matrix_np = np.array(time_matrix, dtype=float)

    beam_width = get_beam_width(NUM_NODES)
    print(f"beam_width (det) = {beam_width}")

    rng = np.random.default_rng(SEED)
    day_indices = np.array([
        rng.choice(num_days_eval, size=N_DAYS_PER_NODE, replace=False)
        for _ in range(NUM_NODES)
    ])

    variants = [("baseline", None)] + [(f"vc={vc}", vc) for vc in VALUE_COEFS]

    rows = []
    for s in range(NUM_NODES):
        for rep in range(N_DAYS_PER_NODE):
            day_idx = int(day_indices[s, rep])

            baseline_reward = None
            for name, vc in variants:
                route, _, _ = agent.beam_search_dynamic(
                    s, day_idx,
                    time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
                    MAX_DURATION,
                    beam_width=beam_width,
                    ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                    avail_prob_arr=None,
                    critic=(critic if vc is not None else None),
                    value_coef=(vc if vc is not None else 1.0),
                )
                valid = route is not None
                if valid:
                    reward, _ = simulate_route_reward(
                        route, s, day_idx,
                        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
                        avail_prob_arr=None,
                    )
                else:
                    reward = -np.inf

                if name == "baseline":
                    baseline_reward = reward

                rows.append({
                    "start_node": s,
                    "rep":        rep,
                    "day_idx":    day_idx,
                    "variant":    name,
                    "reward":     reward,
                    "valid":      valid,
                    "delta_vs_baseline": (reward - baseline_reward) if valid and np.isfinite(baseline_reward) else np.nan,
                })

            print(f"start {s:2d} rep {rep} day {day_idx:2d} | "
                  + " | ".join(f"{name}={rows[-len(variants)+i]['reward']:.0f}"
                                for i, (name, _) in enumerate(variants)),
                  flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cwd, OUTPUT_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado en {csv_path}")

    print("\n" + "=" * 70)
    print(f"RESUMEN — {NUM_NODES} start nodes x {N_DAYS_PER_NODE} reps = {NUM_NODES*N_DAYS_PER_NODE} rutas")
    print("=" * 70)
    print(f"{'Variant':10s} {'AvgReward':>12s} {'AvgDelta':>10s} {'WinRate%':>9s} {'LoseRate%':>10s} {'ValidRate':>10s}")
    for name, _ in variants:
        sub = df[df["variant"] == name]
        avg_reward = sub["reward"].replace(-np.inf, np.nan).mean()
        avg_delta  = sub["delta_vs_baseline"].mean()
        win_rate   = (sub["delta_vs_baseline"] > 0).mean() * 100
        lose_rate  = (sub["delta_vs_baseline"] < 0).mean() * 100
        valid_rate = sub["valid"].mean() * 100
        print(f"{name:10s} {avg_reward:12.1f} {avg_delta:10.2f} {win_rate:9.1f} {lose_rate:10.1f} {valid_rate:10.1f}")

    return df


if __name__ == "__main__":
    main()
