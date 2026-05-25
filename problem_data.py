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
    BIG_M_PENALTY,
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
    cost       = distance_arr * (diesel_arr / MPG) + 163 + 1.2 * distance_arr
    reward_arr = np.round(revenue - cost, 0)

    reward_matrix = pd.DataFrame(reward_arr)

    penalized_arr = reward_arr.copy()
    np.fill_diagonal(penalized_arr, BIG_M_PENALTY)
    reward_matrix_penalized = pd.DataFrame(penalized_arr)

    return reward_matrix, reward_matrix_penalized


def load_matrices(num_nodes: int) -> tuple:
    """Load data files and return multi-day stacks for rate/loads.

    Fixed matrices (time, distance, fuel) are loaded from single CSV files.
    Rate and loads are loaded from .npy stacks covering the last 90 days
    so the training loop can sample one day per episode.

    Expected file shapes
    --------------------
    rate_multiday.npy  : [num_days, N, N]
    load_multiday.npy  : [num_days, N, N]

    Returns
    -------
    time_matrix   : pd.DataFrame  (num_nodes × num_nodes)
    rate_stack    : np.ndarray    (num_days × num_nodes × num_nodes)
    loads_stack   : np.ndarray    (num_days × num_nodes × num_nodes)
    distance_arr  : np.ndarray    (num_nodes × num_nodes)
    diesel_arr    : np.ndarray    (num_nodes × num_nodes)
    noise_sigma   : float
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    # ── Fixed single-snapshot matrices ────────────────────────────
    time_matrix_raw = pd.read_csv(os.path.join(cwd, "duration_matrix.csv"), index_col=0)
    distance_raw    = pd.read_csv(os.path.join(cwd, "distance_matrix.csv"), index_col=0)
    diesel_raw      = pd.read_csv(os.path.join(cwd, "fuel_matrix.csv"),     index_col=0)

    # ── Multi-day stacks — shape: [num_days, N, N] ───────────────
    rate_stack_raw  = np.load(os.path.join(cwd, "rate_multiday.npy"))
    loads_stack_raw = np.load(os.path.join(cwd, "load_multiday.npy"))

    # ── Slice fixed matrices to num_nodes ─────────────────────────
    time_matrix  = (time_matrix_raw.iloc[:num_nodes, :num_nodes]).copy()
    time_matrix.index   = range(num_nodes)   # reset a enteros 0-based
    time_matrix.columns = range(num_nodes)
    distance_arr = distance_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)
    diesel_arr   = diesel_raw.iloc[:num_nodes, :num_nodes].to_numpy(dtype=float)

    # ── Slice stacks to num_nodes (days axis untouched) ──────────
    rate_stack  = rate_stack_raw[:, :num_nodes, :num_nodes].astype(float)
    loads_stack = loads_stack_raw[:, :num_nodes, :num_nodes].astype(float)

    num_days = rate_stack.shape[0]

    # ── Noise sigma from reward variance across all historical days ─
    all_rewards = []
    for d in range(num_days):
        revenue = (rate_stack[d] * distance_arr).copy()
        revenue[loads_stack[d] <= 1] = 0
        cost = distance_arr * (diesel_arr / MPG) + 163 + 1.2 * distance_arr
        all_rewards.append((revenue - cost).flatten())
    all_rewards = np.concatenate(all_rewards)
    reward_std  = np.std(all_rewards)
    noise_sigma = NOISE_FRACTION * reward_std if STOCHASTIC_MODE else 0.0

    print(f"Multi-day data  : {num_days} días cargados para rate y loads")
    print(f"Stochastic mode : {STOCHASTIC_MODE} | "
          f"Noise sigma: {noise_sigma:.1f} raw units "
          f"({NOISE_FRACTION*100:.0f}% of std {reward_std:.1f})")

    return time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, noise_sigma
