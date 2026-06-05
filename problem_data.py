"""
problem_data.py
==============
Data loading, matrix computation, and stochastic reward sampling.

load_matrices(num_nodes)
    → time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, noise_sigma

build_day_matrices(rate_day, loads_day, distance_arr, diesel_arr, num_nodes)
    → reward_matrix, reward_matrix_penalized

sample_stochastic_reward(...) → float

rate_stack  and loads_stack are 3-D numpy arrays shaped [num_days, num_nodes, num_nodes].
A random day is sampled once per episode in training/tuning so the agent learns
from historical variability rather than a single reward distribution snapshot.
"""

import os
import numpy as np
import pandas as pd

from config import (
    MPG,
    STOCHASTIC_MODE, NOISE_FRACTION,
    BIG_M_PENALTY, MARGINAL_COST_SIN_DIESEL,
    TRAIN_DAYS,
)


def sample_stochastic_reward(
    expected_reward: float,
    sigma:           float,
    scale_factor:    float,
) -> float:
    """Return a noisy realised reward (Gaussian perturbation)."""
    noise    = np.random.normal(0, sigma)
    realized = expected_reward + noise
    return realized / scale_factor


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
    reward_matrix            : pd.DataFrame (num_nodes × num_nodes)
    reward_matrix_penalized  : pd.DataFrame (diagonal = BIG_M_PENALTY)
    """
    revenue = (rate_day * distance_arr).copy()
    revenue[loads_day <= 1] = 0
    #cost       = distance_arr * (diesel_arr / MPG) + 158 + 1.2 * distance_arr
    cost       = distance_arr * (diesel_arr / MPG) + distance_arr * MARGINAL_COST_SIN_DIESEL
    reward_arr = np.round(revenue - cost, 0)

    reward_matrix = pd.DataFrame(reward_arr)

    penalized_arr = reward_arr.copy()
    np.fill_diagonal(penalized_arr, BIG_M_PENALTY)
    reward_matrix_penalized = pd.DataFrame(penalized_arr)

    return reward_matrix, reward_matrix_penalized


def load_matrices(num_nodes: int) -> tuple:
    """Load data files and return multi-day stacks for rate/loads.

    Fixed matrices (time, distance, fuel) are loaded from single CSV files.
    Rate and loads are loaded from .npy stacks (datos/) so the training loop
    can sample one day per episode.

    Expected file shapes
    --------------------
    datos/rate.npy             : [num_days, N, N]
    datos/load_availability.npy: [num_days, N, N]

    Returns
    -------3
    time_matrix   : pd.DataFrame  (num_nodes × num_nodes)
    rate_stack    : np.ndarray    (num_days × num_nodes × num_nodes)
    loads_stack   : np.ndarray    (num_days × num_nodes × num_nodes)
    distance_arr  : np.ndarray    (num_nodes × num_nodes)
    diesel_arr    : np.ndarray    (num_nodes × num_nodes)
    noise_sigma   : float
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

    # ── Slice fixed matrices to num_nodes ─────────────────────────
    time_matrix  = (time_matrix_raw.iloc[:num_nodes, :num_nodes]).copy()
    time_matrix.index   = range(num_nodes)   # reset a enteros 0-based
    time_matrix.columns = range(num_nodes)
    distance_arr = distance_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    diesel_arr   = diesel_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)

    # ── Slice stacks to num_nodes; align days to the shorter stack ──
    num_days    = min(rate_stack_raw.shape[0], loads_stack_raw.shape[0])
    rate_stack  = rate_stack_raw[:num_days, :num_nodes, :num_nodes].astype(float)
    loads_stack = loads_stack_raw[:num_days, :num_nodes, :num_nodes].astype(float)

    # ── Noise sigma — computed on training days only ─────────────────
    train_days = min(TRAIN_DAYS, num_days)
    daily_stds = []
    for d in range(train_days):
        revenue = (rate_stack[d] * distance_arr).copy()
        revenue[loads_stack[d] <= 1] = 0
        cost = distance_arr * (diesel_arr / MPG) + distance_arr * MARGINAL_COST_SIN_DIESEL
        rewards_day = (revenue - cost).flatten()
        daily_stds.append(np.std(rewards_day))
    reward_std  = np.mean(daily_stds)
    noise_sigma = NOISE_FRACTION * reward_std if STOCHASTIC_MODE else 0.0

    eval_days = num_days - train_days
    print(f"Multi-day data  : {num_days} días totales | train={train_days} | eval={eval_days}")
    print(f"Stochastic mode : {STOCHASTIC_MODE} | "
        f"Noise sigma: {noise_sigma:.1f} raw units "
        f"({NOISE_FRACTION*100:.0f}% of intra-day std {reward_std:.1f})")

    return time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, noise_sigma



