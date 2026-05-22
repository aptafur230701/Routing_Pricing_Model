"""
environment.py
==============
Data loading, matrix computation, and stochastic reward sampling.


----------
load_matrices(num_nodes)      → time_matrix, reward_matrix,
                                reward_matrix_penalized, noise_sigma
sample_stochastic_reward(...) → float
"""

import os
import numpy as np
import pandas as pd

from config import (
    TRUCK_TYPE, MPG, DATE_SUFFIX, REGION,
    STOCHASTIC_MODE, NOISE_FRACTION,
    BIG_M_PENALTY, REWARD_SCALE_FACTOR,
)


def sample_stochastic_reward(
    expected_reward: float,
    sigma:           float,
    scale_factor:    float,
) -> float:
    """Return a noisy realised reward.

    Models rate fluctuations, load cancellations, and fuel-price variance
    in real TL spot freight markets.
    """
    noise    = np.random.normal(0, sigma)
    realized = expected_reward + noise
    return realized / scale_factor


def load_matrices(num_nodes: int):
    """Load raw CSVs, build reward matrix, slice to num_nodes.

    Returns
    -------
    time_matrix              : pd.DataFrame  (num_nodes × num_nodes)
    reward_matrix            : pd.DataFrame  (num_nodes × num_nodes)
    reward_matrix_penalized  : pd.DataFrame  (diagonal = BIG_M_PENALTY)
    noise_sigma              : float         (0 when STOCHASTIC_MODE=False)
    """
    cwd = os.path.dirname(os.path.abspath(__file__))

    rate_matrix     = pd.read_csv(os.path.join(cwd, f"rate_q2_{REGION}_{TRUCK_TYPE}{DATE_SUFFIX}.csv"),    header=None)
    time_matrix_raw = pd.read_csv(os.path.join(cwd, f"duration_{REGION}.csv"),                            header=None)
    loads_matrix    = pd.read_csv(os.path.join(cwd, f"load_av_{REGION}_{TRUCK_TYPE}{DATE_SUFFIX}.csv"),   header=None)
    distance_matrix = pd.read_csv(os.path.join(cwd, f"distance_{REGION}.csv"),                            header=None)
    diesel_matrix   = pd.read_csv(os.path.join(cwd, f"diesel_{REGION}{DATE_SUFFIX}.csv"),                 header=None)

    # Revenue and cost
    revenue_matrix               = rate_matrix * distance_matrix
    revenue_matrix[loads_matrix <= 1] = 0
    diesel_matrix                = diesel_matrix / MPG
    var_cost_matrix              = 1.2 * distance_matrix
    cost_matrix                  = distance_matrix * diesel_matrix + 163 + var_cost_matrix
    reward_matrix_full           = revenue_matrix - cost_matrix
    time_matrix_full             = time_matrix_raw * 0.9

    # Slice
    time_matrix   = np.round(time_matrix_full.iloc[:num_nodes, :num_nodes],   1)
    reward_matrix = np.round(reward_matrix_full.iloc[:num_nodes, :num_nodes], 0)

    # Stochastic noise sigma
    reward_vals  = reward_matrix.values.flatten()
    reward_std   = np.std(reward_vals)
    noise_sigma  = NOISE_FRACTION * reward_std if STOCHASTIC_MODE else 0.0

    print(f"Stochastic mode: {STOCHASTIC_MODE} | "
          f"Noise sigma: {noise_sigma:.1f} raw units "
          f"({NOISE_FRACTION*100:.0f}% of std {reward_std:.1f})")

    # Penalised matrix (diagonal = BIG_M)
    reward_matrix_penalized = reward_matrix.copy()
    arr = reward_matrix_penalized.to_numpy()
    np.fill_diagonal(arr, BIG_M_PENALTY)
    reward_matrix_penalized = pd.DataFrame(
        arr,
        index=reward_matrix_penalized.index,
        columns=reward_matrix_penalized.columns,
    )

    return time_matrix, reward_matrix, reward_matrix_penalized, noise_sigma
