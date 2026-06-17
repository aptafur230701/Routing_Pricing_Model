"""
problem_data.py
==============
Data loading and matrix computation.

load_matrices(num_nodes)
    → time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
      ltr_stack, trucks_stack, avail_prob_arr

build_day_matrices(rate_day, loads_day, distance_arr, diesel_arr, num_nodes)
    → reward_matrix, reward_matrix_penalized

rate_stack  and loads_stack are 3-D numpy arrays shaped [num_days, num_nodes, num_nodes].
A random day is sampled once per episode in training/tuning so the agent learns
from historical variability rather than a single reward distribution snapshot.
"""

import os
import numpy as np
import pandas as pd

from config import (
    MPG,
    BIG_M_PENALTY, MARGINAL_COST_SIN_DIESEL,
    TRAIN_DAYS,
)


def build_day_matrices(
    rate_day:     np.ndarray,
    loads_day:    np.ndarray,
    distance_arr: np.ndarray,
    diesel_arr:   np.ndarray,
) -> tuple:
    """Build reward matrix and its penalized version for a single day snapshot.

    Parameters
    ----------
    rate_day      : np.ndarray (num_nodes × num_nodes) — spot rates for one day.
    loads_day     : np.ndarray (num_nodes × num_nodes) — available loads for one day.
    distance_arr  : np.ndarray (num_nodes × num_nodes) — inter-node distances.
    diesel_arr    : np.ndarray (num_nodes × num_nodes) — fuel prices.

    Returns
    -------
    reward_matrix            : np.ndarray (num_nodes × num_nodes)
    reward_matrix_penalized  : np.ndarray (diagonal = BIG_M_PENALTY)
    """
    revenue = (rate_day * distance_arr).copy()
    revenue[loads_day <= 1] = 0
    cost       = distance_arr * (diesel_arr / MPG) + distance_arr * MARGINAL_COST_SIN_DIESEL
    reward_arr = np.round(revenue - cost, 0)

    penalized_arr = reward_arr.copy()
    np.fill_diagonal(penalized_arr, BIG_M_PENALTY)

    return reward_arr, penalized_arr


def draw_lane_availability(
    start_day_idx:  int,
    node:           int,
    arrival_day:    int,
    avail_prob_arr: np.ndarray,
    num_nodes:      int,
) -> np.ndarray:
    """Bernoulli draw for outgoing lanes FROM node, using a seed tied to the
    world state (start_day_idx, node, arrival_day) — NOT to the agent step count.

    Two routes arriving at the same node on the same calendar day always see
    the same lane realisation, so DRL and every baseline compete on identical
    stochastic conditions.

    Parameters
    ----------
    start_day_idx  : episode start day index.
    node           : node at which the truck just arrived.
    arrival_day    : min(start_day_idx + int(time_elapsed // 14), max_day).
    avail_prob_arr : [num_nodes, num_nodes] float32 — historical availability probs.
    num_nodes      : number of nodes in the graph.

    Returns
    -------
    np.ndarray of shape (num_nodes,) int8 — 1 = lane exists, 0 = lane absent.
    """
    seed = int((start_day_idx * 9973 + node * 97 + arrival_day) & 0xFFFFFFFF)
    rng  = np.random.default_rng(seed)
    return (rng.random(num_nodes) < avail_prob_arr[node]).astype(np.int8)


def build_rm_pen_stack(
    rate_stack:   np.ndarray,   # (num_days, N, N)
    loads_stack:  np.ndarray,   # (num_days, N, N)
    distance_arr: np.ndarray,   # (N, N)
    diesel_arr:   np.ndarray,   # (N, N)
) -> np.ndarray:                # (num_days, N, N) float32
    """Precompute the full stack of penalized reward matrices for all training days.

    Returns shape (num_days, N, N) float32 — one penalized reward matrix per day.
    Diagonal of each slice is BIG_M_PENALTY. Used by VectorRoutingEnv.
    """
    return np.stack([
        build_day_matrices(rate_stack[d], loads_stack[d], distance_arr, diesel_arr)[1]
        for d in range(len(rate_stack))
    ]).astype(np.float32)


def load_matrices(num_nodes: int) -> tuple:
    """Load data files and return multi-day stacks for rate/loads.

    Fixed matrices (time, distance, fuel) are loaded from single CSV files.
    Rate and loads are loaded from .npy stacks (datos/) so the training loop
    can sample one day per episode.

    Expected file shapes
    --------------------
    datos/rate.npy               : [num_days, N, N]
    datos/load_availability.npy  : [num_days, N, N]
    datos/ltr.npy                : [194, 120]
    datos/trucks_forward.npy     : [194, 600]  — MultiIndex (fecha, delta), 120 días × 5 deltas

    Returns
    -------
    time_matrix    : np.ndarray    (num_nodes × num_nodes) float64
    rate_stack     : np.ndarray    (num_days × num_nodes × num_nodes)
    loads_stack    : np.ndarray    (num_days × num_nodes × num_nodes)
    distance_arr   : np.ndarray    (num_nodes × num_nodes)
    diesel_arr     : np.ndarray    (num_nodes × num_nodes)
    ltr_stack      : np.ndarray    (num_nodes × 120)
    trucks_stack   : np.ndarray    (num_nodes × 120 × 3)  — solo deltas 1,2,3
    avail_prob_arr : np.ndarray    (num_nodes × num_nodes) float32
                     mean over train days of (loads > 0); prior Bernoulli per arc.
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    # ── Fixed single-snapshot matrices ────────────────────────────
    data_dir = os.path.join(cwd, "datos")
    time_matrix_raw = pd.read_csv(os.path.join(data_dir, "duration.csv"), index_col=0)
    distance_raw    = pd.read_csv(os.path.join(data_dir, "distance.csv"), index_col=0)
    diesel_raw      = pd.read_csv(os.path.join(data_dir, "fuel.csv"),     index_col=0)

    # ── Multi-day stacks — shape: [num_days, N, N] ───────────────
    rate_stack_raw  = np.load(os.path.join(data_dir, "rate.npy"))
    loads_stack_raw = np.load(os.path.join(data_dir, "load_availability.npy"))

    # ── Market signal stacks ──────────────────────────────────────
    # ltr: [194, 120] → slice a [num_nodes, 120]
    ltr_raw = np.load(os.path.join(data_dir, "ltr.npy"))
    ltr_stack = ltr_raw[:num_nodes, :].astype(np.float32)

    # trucks_forward: [194, 600] donde 600 = 120 fechas × 5 deltas (orden: fecha es nivel outer)
    # reshape a [num_nodes, 120, 5] y tomar solo deltas 1,2,3 (índices 0,1,2)
    trucks_raw = np.load(os.path.join(data_dir, "trucks_forward.npy"))
    trucks_stack = trucks_raw[:num_nodes, :].astype(np.float32).reshape(num_nodes, 120, 5)[:, :, 0:3]

    # ── Slice fixed matrices to num_nodes ─────────────────────────
    time_matrix  = time_matrix_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    distance_arr = distance_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    diesel_arr   = diesel_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)

    # ── Slice stacks to num_nodes; align days to the shorter stack ──
    num_days    = min(rate_stack_raw.shape[0], loads_stack_raw.shape[0])
    rate_stack  = rate_stack_raw[:num_days, :num_nodes, :num_nodes].astype(float)
    loads_stack = loads_stack_raw[:num_days, :num_nodes, :num_nodes].astype(float)

    # ── Availability prior: P(lane i→j exists on a given day) ────────────────
    # Computed over the full historical window (train + eval) so the prior is
    # the same regardless of how many days are split off for training.
    avail_prob_arr = (loads_stack > 0).mean(axis=0).astype(np.float32)

    train_days = min(TRAIN_DAYS, num_days)
    eval_days  = num_days - train_days
    print(f"Multi-day data  : {num_days} días totales | train={train_days} | eval={eval_days}")

    # P95 de los valores positivos del rate_stack de entrenamiento.
    # Usado como normalizador global de rewards en la feature [7] del encoder.
    rate_train_slice = rate_stack[:train_days]
    _pos_vals = rate_train_slice[rate_train_slice > 0].ravel()
    reward_global_p95 = float(np.percentile(_pos_vals, 95)) if len(_pos_vals) > 0 else 1.0
    print(f"REWARD_GLOBAL_P95: {reward_global_p95:.4f}")

    return time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
           ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95



