"""
sweep_decode_stochastic_35nodes.py
====================================
Gemelo estocástico de sweep_decode_35nodes.py. Barre beam_width sobre el
checkpoint ya entrenado de 35 nodos, evaluado en el MUNDO ESTOCÁSTICO pero
fijo por semilla: la disponibilidad de lanes se sortea con
draw_lane_availability(start_day_idx, node, arrival_day, ...), cuya semilla
es función determinista de (start_day_idx, node, arrival_day). Por tanto,
para un (start_node, day_idx) dado, todas las estrategias compiten sobre
exactamente la misma realización del mundo — solo cambia cómo el agente la
explora (beam_width). No se reentrena nada.

Replica EXACTAMENTE el esquema de evaluation.py::run_solver_comparison:
  - rng = np.random.default_rng(SEED)
  - day_indices[s, rep] = rng.choice(num_days_eval, size=n_days_per_node,
    replace=False) por cada start node — MISMOS días que el barrido
    determinista, para que ambos sean comparables.

Cotas/baselines por (start_node, rep), calculadas UNA sola vez y cacheadas:
  - LS-Oracle      : cota clarividente (Solvers.solve_label_setting_oracle).
  - RH-Lookahead Real : mejor baseline no-clarividente (techo realista).
  - MC-Rollout     : Monte Carlo rollout estocástico.

El reward de cada estrategia DRL se recalcula con Solvers.simulate_route_reward
usando la MISMA semilla Bernoulli (avail_prob_arr activo) con la que se
construyó la ruta, para paridad bit-exacta — igual que el patrón ya usado en
sweep_decode_35nodes.py para el mundo determinista.

Esta variante barre una rejilla fina de beam_width (1,2,3,5,7,10) para ubicar
la "rodilla" de la curva gap-vs-beam en el mundo estocástico, igual que la
rejilla fina de sweep_decode_35nodes.py para el mundo determinista. Las cotas
(LS-Oracle, RH-Lookahead Real, MC-Rollout) se reutilizan del CSV de la corrida
anterior (sweep_decode_stochastic_35nodes.csv) cuando está disponible, para no
volver a pagar 105 resoluciones de label-setting / rolling-horizon / MC-rollout.
"""

import os
import time

import numpy as np
import pandas as pd
import torch

from config import (
    SEED, DEVICE, TRAIN_DAYS, MAX_DURATION,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
)
from problem_data import load_matrices
from am_agent import AMRoutingAgent
from Solvers import (
    simulate_route_reward,
    solve_label_setting_oracle,
    solve_heuristic_rolling_horizon_lookahead_stochastic,
    solve_mc_rollout_stochastic,
)

NUM_NODES       = 35
N_DAYS_PER_NODE = 3
ORACLE_TIME_LIMIT_SECONDS = 300
LOOKAHEAD = 3
ASSERT_TOL = 1e-3
BOUNDS_CACHE_CSV   = "sweep_decode_stochastic_35nodes.csv"   # corrida previa — reutilizable
OUTPUT_CSV         = "sweep_decode_stochastic_35nodes_fine.csv"
KNEE_THRESHOLD_PCT = 2.0   # puntos de gap recuperados por debajo de este umbral → rodilla

STRATEGIES = [
    ("real_beam1",  1),
    ("real_beam2",  2),
    ("real_beam3",  3),
    ("real_beam5",  5),
    ("real_beam7",  7),
    ("real_beam10", 10),
]


def load_bounds_cache(cwd: str) -> dict:
    """Reutiliza oracle_reward/rhlr_reward/mc_reward de la corrida previa
    (sweep_decode_stochastic_35nodes.csv), indexado por (start_node, rep).
    Evita re-resolver LS-Oracle/RH-Lookahead-Real/MC-Rollout (caros) para
    días/nodos ya resueltos."""
    cache_path = os.path.join(cwd, BOUNDS_CACHE_CSV)
    if not os.path.exists(cache_path):
        print(f"Aviso: no se encontró {cache_path} — se recalcularán las cotas para todo.")
        return {}
    prev = pd.read_csv(cache_path)
    dedup = prev[["start_node", "rep", "oracle_reward", "rhlr_reward", "mc_reward"]] \
        .drop_duplicates(subset=["start_node", "rep"])
    cache = {
        (int(r.start_node), int(r.rep)): (
            float(r.oracle_reward), float(r.rhlr_reward), float(r.mc_reward)
        )
        for r in dedup.itertuples()
    }
    print(f"Cache de cotas cargado desde {cache_path}: {len(cache)} entradas (start_node, rep).")
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

    bounds_cache = load_bounds_cache(cwd)

    # MISMO esquema de RNG que evaluation.py::run_solver_comparison y que
    # sweep_decode_35nodes.py — produce los mismos (start_node, rep) -> day_idx.
    rng = np.random.default_rng(SEED)
    day_indices = np.array([
        rng.choice(num_days_eval, size=n_days_per_node, replace=False)
        for _ in range(NUM_NODES)
    ])

    start_nodes = (
        start_nodes_subset if start_nodes_subset is not None else range(NUM_NODES)
    )

    rows = []
    strat_rewards = {name: [] for name, _ in STRATEGIES}
    strat_gaps    = {name: [] for name, _ in STRATEGIES}
    strat_adv     = {name: [] for name, _ in STRATEGIES}
    strat_valid   = {name: [] for name, _ in STRATEGIES}
    strat_time    = {name: 0.0 for name, _ in STRATEGIES}
    rhlr_rewards  = []
    mc_rewards    = []

    for s in start_nodes:
        for rep in range(n_days_per_node):
            day_idx = int(day_indices[s, rep])
            print(f"\n[{label}] Start node {s} | rep {rep} | eval day {day_idx} "
                  f"(abs day {TRAIN_DAYS + day_idx})", flush=True)

            cached_bounds = bounds_cache.get((s, rep))
            if cached_bounds is not None:
                oracle_reward, rhlr_reward, mc_reward = cached_bounds
                print(f"  [cached] LS-Oracle: {oracle_reward:.1f} | "
                      f"RH-Lookahead Real: {rhlr_reward:.1f} | MC-Rollout: {mc_reward:.1f}")
            else:
                # ── LS-Oracle (cota clarividente) — una sola vez por (s, rep) ──
                t0 = time.time()
                oracle_status, oracle_route, oracle_reward, oracle_duration = solve_label_setting_oracle(
                    s, time_matrix_np, rate_eval, loads_eval,
                    distance_arr, diesel_arr, MAX_DURATION, NUM_NODES, day_idx,
                    avail_prob_arr=avail_prob_arr,
                    time_limit_seconds=ORACLE_TIME_LIMIT_SECONDS,
                )
                oracle_time = time.time() - t0
                print(f"  LS-Oracle [{oracle_status}]: reward {oracle_reward:.1f} ({oracle_time:.1f}s)")

                # ── RH-Lookahead Real — mejor baseline no clarividente ──────────
                rhlr_status, rhlr_route, rhlr_reward, rhlr_duration, rhlr_valid = \
                    solve_heuristic_rolling_horizon_lookahead_stochastic(
                        s, time_matrix, rate_eval, loads_eval,
                        distance_arr, diesel_arr, MAX_DURATION, NUM_NODES, day_idx,
                        avail_prob_arr=avail_prob_arr,
                        lookahead=LOOKAHEAD,
                    )
                rhlr_reward = rhlr_reward if rhlr_valid else np.nan
                print(f"  RH-Lookahead Real [{rhlr_status}]: reward {rhlr_reward:.1f}")

                # ── MC-Rollout ───────────────────────────────────────────────
                mc_status, mc_route, mc_reward, mc_duration, mc_valid = \
                    solve_mc_rollout_stochastic(
                        s, time_matrix, rate_eval, loads_eval,
                        distance_arr, diesel_arr, MAX_DURATION, NUM_NODES, day_idx,
                        avail_prob_arr=avail_prob_arr,
                    )
                mc_reward = mc_reward if mc_valid else np.nan
                print(f"  MC-Rollout [{mc_status}]: reward {mc_reward:.1f}")

            rhlr_rewards.append(rhlr_reward)
            mc_rewards.append(mc_reward)

            for name, beam_width in STRATEGIES:
                t0 = time.time()
                route, _, _ = agent.beam_search_dynamic(
                    s, day_idx,
                    time_matrix, rate_eval, loads_eval, distance_arr, diesel_arr,
                    MAX_DURATION,
                    beam_width=beam_width,
                    ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                    avail_prob_arr=avail_prob_arr,
                    reward_global_p95=reward_global_p95,
                )
                rollout_time = time.time() - t0
                strat_time[name] += rollout_time

                valid = route is not None
                if valid:
                    reward, _ = simulate_route_reward(
                        route, s, day_idx,
                        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
                        avail_prob_arr=avail_prob_arr,
                    )
                else:
                    reward = -np.inf

                # Sanity check 2: la cota clarividente nunca puede ser superada.
                if valid and np.isfinite(oracle_reward):
                    assert oracle_reward >= reward - ASSERT_TOL, (
                        f"LS-Oracle ({oracle_reward:.1f}) superado por {name} "
                        f"({reward:.1f}) en start={s} day={day_idx} — "
                        f"desalineación de semillas Bernoulli."
                    )

                if np.isfinite(oracle_reward) and oracle_reward != 0:
                    gap_pct = (oracle_reward - reward) / oracle_reward * 100 if valid else np.inf
                else:
                    gap_pct = np.nan

                if valid and np.isfinite(rhlr_reward) and rhlr_reward != 0:
                    adv_pct = (reward - rhlr_reward) / rhlr_reward * 100
                else:
                    adv_pct = np.nan

                strat_rewards[name].append(reward if valid else np.nan)
                strat_gaps[name].append(gap_pct)
                strat_adv[name].append(adv_pct)
                strat_valid[name].append(valid)

                print(f"    {name:12s} reward {reward:10.1f} | gap_oracle {gap_pct:6.2f}% "
                      f"| adv_vs_rhlr {adv_pct:6.2f}% | valid={valid} | {rollout_time:.2f}s")

                rows.append({
                    "start_node":     s,
                    "rep":            rep,
                    "strategy":       name,
                    "reward":         reward,
                    "oracle_reward":  oracle_reward,
                    "gap_oracle_pct": gap_pct,
                    "rhlr_reward":    rhlr_reward,
                    "adv_vs_rhlr_pct": adv_pct,
                    "mc_reward":      mc_reward,
                    "valid":          valid,
                    "rollout_time":   rollout_time,
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(cwd, OUTPUT_CSV)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV guardado en {csv_path}")

    n_routes = len(list(start_nodes)) * n_days_per_node

    avg_rhlr = np.nanmean(np.array(rhlr_rewards, dtype=float))
    avg_mc   = np.nanmean(np.array(mc_rewards, dtype=float))

    print("\n" + "=" * 90)
    print(f"RESUMEN [{label}] — {len(list(start_nodes))} start nodes x {n_days_per_node} reps")
    print("=" * 90)
    print(f"Referencias fijas — RH-Lookahead Real avg reward: {avg_rhlr:.1f} | "
          f"MC-Rollout avg reward: {avg_mc:.1f}")
    print(f"{'Strategy':12s} {'AvgReward':>12s} {'AvgGap_Oracle%':>15s} "
          f"{'Adv_vs_RHLR%':>13s} {'ValidRate':>10s} {'TotalTime(s)':>14s} {'Time/route(s)':>14s}")
    avg_gap_by_name = {}
    time_per_route_by_name = {}
    for name, _ in STRATEGIES:
        rewards = np.array(strat_rewards[name], dtype=float)
        gaps    = np.array(strat_gaps[name], dtype=float)
        advs    = np.array(strat_adv[name], dtype=float)
        valids  = np.array(strat_valid[name], dtype=bool)
        avg_reward = np.nanmean(rewards) if len(rewards) else np.nan
        avg_gap    = np.nanmean(gaps[np.isfinite(gaps)]) if np.isfinite(gaps).any() else np.nan
        avg_adv    = np.nanmean(advs[np.isfinite(advs)]) if np.isfinite(advs).any() else np.nan
        valid_rate = valids.mean() if len(valids) else np.nan
        time_per_route = strat_time[name] / n_routes if n_routes else np.nan
        avg_gap_by_name[name] = avg_gap
        time_per_route_by_name[name] = time_per_route
        print(f"{name:12s} {avg_reward:12.1f} {avg_gap:15.2f} {avg_adv:13.2f} "
              f"{valid_rate:10.2%} {strat_time[name]:14.2f} {time_per_route:14.4f}")

    rb1_rewards = np.array(strat_rewards["real_beam1"], dtype=float)
    rb1_gaps    = np.array(strat_gaps["real_beam1"], dtype=float)
    print(f"\nSanity check 1 — real_beam1 avg reward: {np.nanmean(rb1_rewards):.1f} "
          f"(esperado ~3275), gap vs LS-Oracle: "
          f"{np.nanmean(rb1_gaps[np.isfinite(rb1_gaps)]):.2f}% (esperado ~30.8%)")
    print("Sanity check 2 — LS-Oracle >= cada estrategia DRL: verificado vía assert "
          "en cada (start_node, rep, strategy) durante el loop (no hubo AssertionError).")

    # ── Análisis de rodilla ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANÁLISIS DE RODILLA (mundo estocástico)")
    print("=" * 70)
    names = [name for name, _ in STRATEGIES]
    knee_name = None
    for prev_name, next_name in zip(names[:-1], names[1:]):
        gap_recovered = avg_gap_by_name[prev_name] - avg_gap_by_name[next_name]
        time_increase = time_per_route_by_name[next_name] - time_per_route_by_name[prev_name]
        efficiency = gap_recovered / time_increase if time_increase > 0 else np.nan
        print(f"  {prev_name:11s} -> {next_name:11s} | gap recuperado: {gap_recovered:6.2f} pts | "
              f"+tiempo/ruta: {time_increase:7.4f}s | eficiencia: {efficiency:8.2f} pts/s")
        if knee_name is None and gap_recovered < KNEE_THRESHOLD_PCT:
            knee_name = prev_name

    if knee_name is None:
        knee_name = names[-1]   # ningún salto cayó debajo del umbral — la rodilla es el último beam probado

    print(f"\nRODILLA sugerida (estocástico): beam={knee_name.replace('real_beam', '')} "
          f"(gap_oracle {avg_gap_by_name[knee_name]:.2f}%, "
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
