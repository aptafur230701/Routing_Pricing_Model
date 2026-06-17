"""
benchmark_hotpath.py
====================
Microbenchmark: old (loop-based) vs new (vectorized) implementations of the
three hottest functions. Runs each version 1000× on identical inputs.

Usage: python3 benchmark_hotpath.py
"""

import math, time
import numpy as np

TRUCKS_CLIP = 50.0
N   = 20
RNG = np.random.default_rng(42)

reward_mat = (RNG.standard_normal((N, N)) * 200).astype(np.float64)
np.fill_diagonal(reward_mat, -1e9)
time_mat   = RNG.uniform(5.0, 60.0, (N, N)).astype(np.float64)
dist_mat   = RNG.uniform(100.0, 1000.0, (N, N)).astype(np.float64)
trucks_s   = RNG.uniform(0, 45, (N, 3)).astype(np.float32)
avail_p    = RNG.uniform(0.0, 1.0, (N, N)).astype(np.float32)
lane_ex    = (RNG.uniform(0, 1, N) > 0.3).astype(np.int8)

cnode, snode  = 5, 0
visited_set   = {0, 3, 5}
time_elapsed  = 20.0
MAX_DUR       = 77.0
P95           = 500.0
REPS          = 2000

# ── OLD build_node_features (loop) ────────────────────────────────────────────
def build_node_features_old(current_node, start_node, visited_set,
                             reward_matrix_penalized, time_matrix,
                             distance_arr, num_nodes, max_duration,
                             trucks_stack=None, time_matrix_arr=None,
                             avail_prob_arr=None, reward_global_p95=1.0):
    N_FEAT = 8
    feats = np.zeros((num_nodes, N_FEAT), dtype=np.float32)
    for j in range(num_nodes):
        feats[j, 0] = float(reward_matrix_penalized[current_node][j])
        feats[j, 1] = float(time_matrix[current_node][j])
        feats[j, 2] = float(distance_arr[current_node, j])
        feats[j, 3] = 1.0 if j == start_node else 0.0
        feats[j, 4] = 1.0 if j in visited_set else 0.0
        if trucks_stack is not None and time_matrix_arr is not None:
            travel_hours = float(time_matrix_arr[current_node, j])
            delta_idx = max(0, min(2, math.ceil(travel_hours / 14.0) - 1))
            trucks_val = float(trucks_stack[j, delta_idx])
            feats[j, 5] = min(trucks_val, TRUCKS_CLIP) / TRUCKS_CLIP
        if avail_prob_arr is not None:
            feats[j, 6] = float(avail_prob_arr[current_node, j])
    feats[:, 0] = np.clip(feats[:, 0], -1e4, 1e4)
    raw_rewards = feats[:, 0].copy()
    for col in (0, 2):
        col_min, col_max = feats[:, col].min(), feats[:, col].max()
        if col_max > col_min:
            feats[:, col] = (feats[:, col] - col_min) / (col_max - col_min)
        else:
            feats[:, col] = 0.0
    feats[:, 1] = np.clip(feats[:, 1] / max(max_duration, 1e-6), 0.0, 1.0)
    _d = max(reward_global_p95, 1e-6)
    feat7 = np.clip(raw_rewards / _d, 0.0, 1.0)
    feat7[raw_rewards < 0] = 0.0
    feats[:, 7] = feat7
    return feats

# ── NEW build_node_features (vectorized) ──────────────────────────────────────
from attention_encoder import build_node_features as build_node_features_new

# ── OLD _get_action_mask (loops) ──────────────────────────────────────────────
def get_action_mask_old(time_matrix, num_nodes, current_node, start_node,
                         visited_set, time_elapsed, max_duration,
                         current_lane_exists=None):
    mask = np.ones(num_nodes, dtype=np.int8)
    mask[current_node] = 0
    for v in visited_set:
        if v != start_node:
            mask[v] = 0
    for j in range(num_nodes):
        if mask[j] == 1 and j != start_node:
            t_to_j     = float(time_matrix[current_node][j])
            t_j_to_dep = float(time_matrix[j][start_node])
            if time_elapsed + t_to_j + t_j_to_dep > max_duration:
                mask[j] = 0
    if current_lane_exists is not None and current_node != start_node:
        for j in range(num_nodes):
            if mask[j] == 1 and j != start_node:
                if current_lane_exists[j] == 0:
                    mask[j] = 0
    return mask

# ── NEW _get_action_mask (vectorized) ─────────────────────────────────────────
def get_action_mask_new(time_matrix, num_nodes, current_node, start_node,
                         visited_set, time_elapsed, max_duration,
                         current_lane_exists=None):
    mask = np.ones(num_nodes, dtype=np.int8)
    mask[current_node] = 0
    visited_inter = [v for v in visited_set if v != start_node]
    if visited_inter:
        mask[visited_inter] = 0
    non_start   = np.arange(num_nodes) != start_node
    t_to_j      = time_matrix[current_node]
    t_j_to_dep  = time_matrix[:, start_node]
    over_budget = (time_elapsed + t_to_j + t_j_to_dep) > max_duration
    mask[non_start & over_budget] = 0
    if current_lane_exists is not None and current_node != start_node:
        lane_absent = (current_lane_exists == 0)
        mask[non_start & lane_absent] = 0
    return mask

# ── Benchmark helper ─────────────────────────────────────────────────────────
def bench(fn, *args, n=REPS, **kwargs):
    # warmup
    for _ in range(10):
        fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / n * 1e6  # µs per call

# ── Run benchmarks ────────────────────────────────────────────────────────────
print(f"Microbenchmark — N={N}, {REPS} reps each\n")

t_old = bench(build_node_features_old,
              cnode, snode, visited_set, reward_mat, time_mat, dist_mat, N, MAX_DUR,
              trucks_stack=trucks_s, time_matrix_arr=time_mat,
              avail_prob_arr=avail_p, reward_global_p95=P95)
t_new = bench(build_node_features_new,
              cnode, snode, visited_set, reward_mat, time_mat, dist_mat, N, MAX_DUR,
              trucks_stack=trucks_s, time_matrix_arr=time_mat,
              avail_prob_arr=avail_p, reward_global_p95=P95)
print(f"build_node_features : old={t_old:6.1f} µs  new={t_new:6.1f} µs  "
      f"speedup={t_old/t_new:.1f}×")

t_old2 = bench(get_action_mask_old,
               time_mat, N, cnode, snode, visited_set, time_elapsed, MAX_DUR, lane_ex)
t_new2 = bench(get_action_mask_new,
               time_mat, N, cnode, snode, visited_set, time_elapsed, MAX_DUR, lane_ex)
print(f"_get_action_mask    : old={t_old2:6.1f} µs  new={t_new2:6.1f} µs  "
      f"speedup={t_old2/t_new2:.1f}×")

# ── Repeat for N=20 extrapolation to real training ────────────────────────────
from problem_data import build_day_matrices
rate_day   = RNG.uniform(0, 10, (N, N))
loads_day  = RNG.uniform(0, 5, (N, N))
diesel_arr = RNG.uniform(3, 7, (N, N))

t_bdm = bench(build_day_matrices, rate_day, loads_day, dist_mat, diesel_arr, n=1000)
print(f"build_day_matrices  : {t_bdm:6.1f} µs  "
      f"(saved on every step via cache; ~{t_bdm:.0f}µs × steps × episodes)")

print("\nNote: savings from build_day_matrices cache depend on num_unique_days×steps.")
print("With max_duration=77h each episode hits 1-6 calendar days, so cache hit rate ≈ 90%+.")
