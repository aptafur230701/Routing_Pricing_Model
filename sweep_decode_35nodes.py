"""
sweep_decode_35nodes.py
========================
Barrido de estrategias de decodificación (beam search vs sampling) sobre el
checkpoint ya entrenado de 35 nodos, evaluado en el set determinista de
evaluación (sin Bernoulli de disponibilidad de lanes). No reentrena nada.

Replica EXACTAMENTE el esquema de evaluation.py::run_solver_comparison:
  - rng = np.random.default_rng(SEED)
  - day_indices[s, rep] = rng.choice(num_days_eval, size=n_days_per_node,
    replace=False) por cada start node, generado ANTES del loop principal.
  - day_idx es relativo al slice de evaluación (rate_stack[TRAIN_DAYS:]).

Las rutas se generan con agent.beam_search_dynamic() (beam) o
agent.generate_route_sampling() (sampling) — ambos con día de mercado
dinámico (day_offset = time_elapsed // 14), igual que el mecanismo real
que produjo el ~48.3% de gap reportado en el summary de DRL Det.

El reward final de cada ruta se recalcula con Solvers.simulate_route_reward
para garantizar paridad bit-exacta con el mundo determinista, igual que hace
evaluation.py para DRL Det. La cota LS-Exact se calcula una sola vez por
(start_node, rep) y se reutiliza para todas las estrategias — y se reutiliza
también del CSV de la corrida anterior (sweep_decode_35nodes.csv) cuando está
disponible, para no volver a pagar 105 resoluciones de label-setting exacto.

Esta variante barre una rejilla fina de beam_width (1,2,3,5,7,10) para ubicar
la "rodilla" de la curva gap-vs-beam: el punto a partir del cual aumentar el
beam deja de comprar mejoras de gap proporcionales al costo de cómputo extra.
"""

import os
import time

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
from Solvers import simulate_route_reward, solve_label_setting_exact

NUM_NODES       = 35
N_DAYS_PER_NODE = 3
LS_EXACT_TIME_LIMIT_SECONDS = 300
LS_EXACT_CACHE_CSV = "sweep_decode_35nodes.csv"   # corrida previa — reutilizable
OUTPUT_CSV         = "sweep_decode_35nodes_fine.csv"
KNEE_THRESHOLD_PCT  = 2.0   # puntos de gap recuperados por debajo de este umbral → rodilla

STRATEGIES = [
    ("beam1",  "beam", 1),
    ("beam2",  "beam", 2),
    ("beam3",  "beam", 3),
    ("beam5",  "beam", 5),
    ("beam7",  "beam", 7),
    ("beam10", "beam", 10),
]


def load_ls_exact_cache(cwd: str) -> dict:
    """Reutiliza ls_exact_reward de la corrida previa (sweep_decode_35nodes.csv),
    indexado por (start_node, rep) → ls_exact_reward. Evita re-resolver
    label-setting exacto (caro) para días/nodos ya resueltos."""
    cache_path = os.path.join(cwd, LS_EXACT_CACHE_CSV)
    if not os.path.exists(cache_path):
        print(f"Aviso: no se encontró {cache_path} — se recalculará LS-Exact para todo.")
        return {}
    prev = pd.read_csv(cache_path)
    dedup = prev[["start_node", "rep", "ls_exact_reward"]].drop_duplicates(
        subset=["start_node", "rep"]
    )
    cache = {
        (int(r.start_node), int(r.rep)): float(r.ls_exact_reward)
        for r in dedup.itertuples()
    }
    print(f"LS-Exact cache cargado desde {cache_path}: {len(cache)} entradas (start_node, rep).")
    return cache


def load_agent(checkpoint_dir: str, num_nodes: int) -> AMRoutingAgent:
    """Misma lógica que main.py::_load_agent — checkpoint ya entrenado, sin reentrenar."""
    path = os.path.join(checkpoint_dir, f"am_checkpoint_{num_nodes}nodes.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {path}")
    ckpt  = torch.load(path, map_location=DEVICE)
    agent = AMRoutingAgent(
        num_nodes, AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, device=DEVICE
    )
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    print(f"Checkpoint cargado: {path}")
    return agent


def run_strategy(agent, strat_kind, strat_param, start_node, day_idx,
                  time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
                  ltr_stack, trucks_stack):
    """Genera una ruta con la estrategia dada. Devuelve (route, rollout_time)."""
    t0 = time.time()
    if strat_kind == "beam":
        route, _, _ = agent.beam_search_dynamic(
            start_node, day_idx,
            time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
            MAX_DURATION,
            beam_width=strat_param,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=None,
        )
    elif strat_kind == "sample":
        route, _, _ = agent.generate_route_sampling(
            start_node, day_idx,
            time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
            n_samples=strat_param,
            max_duration=MAX_DURATION,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=None,
        )
    else:
        raise ValueError(f"Estrategia desconocida: {strat_kind}")
    rollout_time = time.time() - t0
    return route, rollout_time


def main(start_nodes_subset=None, n_days_per_node=N_DAYS_PER_NODE, label="full"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95 = load_matrices(NUM_NODES)

    rate_eval  = rate_stack[TRAIN_DAYS:]
    loads_eval = loads_stack[TRAIN_DAYS:]
    num_days_eval = rate_eval.shape[0]

    time_matrix_np = np.array(time_matrix, dtype=float)

    cwd = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(cwd, "checkpoints")
    agent = load_agent(checkpoint_dir, NUM_NODES)

    ls_exact_cache = load_ls_exact_cache(cwd)

    rng = np.random.default_rng(SEED)
    day_indices = np.array([
        rng.choice(num_days_eval, size=n_days_per_node, replace=False)
        for _ in range(NUM_NODES)
    ])

    start_nodes = (
        start_nodes_subset if start_nodes_subset is not None else range(NUM_NODES)
    )

    rows = []
    strat_rewards = {name: [] for name, _, _ in STRATEGIES}
    strat_gaps    = {name: [] for name, _, _ in STRATEGIES}
    strat_valid   = {name: [] for name, _, _ in STRATEGIES}
    strat_time    = {name: 0.0 for name, _, _ in STRATEGIES}

    for s in start_nodes:
        for rep in range(n_days_per_node):
            day_idx = int(day_indices[s, rep])
            print(f"\n[{label}] Start node {s} | rep {rep} | eval day {day_idx} "
                  f"(abs day {TRAIN_DAYS + day_idx})", flush=True)

            # LS-Exact — una sola vez por (start_node, rep); reutiliza la corrida
            # previa si está cacheada, si no la resuelve (300s de time limit).
            cached_ls_reward = ls_exact_cache.get((s, rep))
            if cached_ls_reward is not None:
                ls_reward = cached_ls_reward
                print(f"  LS-Exact [cached]: reward {ls_reward:.1f}")
            else:
                t0 = time.time()
                ls_status, ls_route, ls_reward, ls_duration = solve_label_setting_exact(
                    s, time_matrix_np, rate_eval, loads_eval,
                    distance_arr, diesel_arr, MAX_DURATION, NUM_NODES, day_idx,
                    avail_prob_arr=None,
                    time_limit_seconds=LS_EXACT_TIME_LIMIT_SECONDS,
                )
                ls_time = time.time() - t0
                print(f"  LS-Exact [{ls_status}]: reward {ls_reward:.1f} ({ls_time:.1f}s)")

            for name, kind, param in STRATEGIES:
                route, rollout_time = run_strategy(
                    agent, kind, param, s, day_idx,
                    time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
                    ltr_stack, trucks_stack,
                )
                strat_time[name] += rollout_time

                valid = route is not None
                if valid:
                    reward, _ = simulate_route_reward(
                        route, s, day_idx,
                        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
                        avail_prob_arr=None,
                    )
                else:
                    reward = -np.inf

                if ls_reward not in (0.0,) and np.isfinite(ls_reward) and ls_reward != 0:
                    gap_pct = (ls_reward - reward) / ls_reward * 100 if valid else np.inf
                else:
                    gap_pct = np.nan

                strat_rewards[name].append(reward if valid else np.nan)
                strat_gaps[name].append(gap_pct)
                strat_valid[name].append(valid)

                print(f"    {name:14s} reward {reward:10.1f} | gap {gap_pct:6.2f}% "
                      f"| valid={valid} | {rollout_time:.2f}s")

                rows.append({
                    "start_node":    s,
                    "rep":           rep,
                    "strategy":      name,
                    "reward":        reward,
                    "ls_exact_reward": ls_reward,
                    "gap_pct":       gap_pct,
                    "valid":         valid,
                    "rollout_time":  rollout_time,
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cwd, OUTPUT_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado en {csv_path}")

    n_routes = len(list(start_nodes)) * n_days_per_node

    print("\n" + "=" * 70)
    print(f"RESUMEN [{label}] — {len(list(start_nodes))} start nodes x {n_days_per_node} reps")
    print("=" * 70)
    print(f"{'Strategy':10s} {'AvgReward':>12s} {'AvgGap%':>10s} {'ValidRate':>10s} "
          f"{'TotalTime(s)':>14s} {'Time/route(s)':>14s}")
    avg_gap_by_name = {}
    time_per_route_by_name = {}
    for name, _, _ in STRATEGIES:
        rewards = np.array(strat_rewards[name], dtype=float)
        gaps    = np.array(strat_gaps[name], dtype=float)
        valids  = np.array(strat_valid[name], dtype=bool)
        avg_reward = np.nanmean(rewards) if len(rewards) else np.nan
        avg_gap    = np.nanmean(gaps[np.isfinite(gaps)]) if np.isfinite(gaps).any() else np.nan
        valid_rate = valids.mean() if len(valids) else np.nan
        time_per_route = strat_time[name] / n_routes if n_routes else np.nan
        avg_gap_by_name[name] = avg_gap
        time_per_route_by_name[name] = time_per_route
        print(f"{name:10s} {avg_reward:12.1f} {avg_gap:10.2f} {valid_rate:10.2%} "
              f"{strat_time[name]:14.2f} {time_per_route:14.4f}")

    baseline_gap = avg_gap_by_name["beam1"]
    print(f"\nSanity check — beam1 gap vs LS-Exact: {baseline_gap:.2f}% "
          f"(esperado ~48.3% según summary previo)")

    # ── Análisis de rodilla ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANÁLISIS DE RODILLA")
    print("=" * 70)
    names = [name for name, _, _ in STRATEGIES]
    knee_name = None
    for prev_name, next_name in zip(names[:-1], names[1:]):
        gap_recovered = avg_gap_by_name[prev_name] - avg_gap_by_name[next_name]
        time_increase = time_per_route_by_name[next_name] - time_per_route_by_name[prev_name]
        efficiency = gap_recovered / time_increase if time_increase > 0 else np.nan
        print(f"  {prev_name:6s} -> {next_name:6s} | gap recuperado: {gap_recovered:6.2f} pts | "
              f"+tiempo/ruta: {time_increase:7.4f}s | eficiencia: {efficiency:8.2f} pts/s")
        if knee_name is None and gap_recovered < KNEE_THRESHOLD_PCT:
            knee_name = prev_name

    if knee_name is None:
        knee_name = names[-1]   # ningún salto cayó debajo del umbral — la rodilla es el último beam probado

    print(f"\nRODILLA sugerida: beam={knee_name.replace('beam', '')} "
          f"(gap {avg_gap_by_name[knee_name]:.2f}%, "
          f"tiempo {time_per_route_by_name[knee_name]:.4f} s/ruta)")

    print("\nNota de escalamiento: el tiempo por ruta anterior se midió a 35 nodos. "
          "A 100 nodos ese tiempo crecerá por tres factores simultáneos — más pasos "
          "por rollout, encoder sobre más nodos, y el propio beam_width — así que el "
          "beam de la rodilla a 100 nodos deberá re-medirse en ese tamaño y probablemente "
          "sea menor que el sugerido aquí. No se extrapola un número.")

    return df


if __name__ == "__main__":
    import sys

    if "--smoke" in sys.argv:
        main(start_nodes_subset=[0, 1], n_days_per_node=N_DAYS_PER_NODE, label="smoke-test")
    else:
        main()
