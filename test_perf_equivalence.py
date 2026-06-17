"""
test_perf_equivalence.py
========================
Validates numerical equivalence of all optimised hot-path functions against
hand-coded reference implementations run on the same random inputs.

Run with:
    python test_perf_equivalence.py

All assertions compare with np.allclose(atol=1e-5) — well within float32 precision.
"""

import math
import numpy as np
import sys
import os

# ── shared RNG ────────────────────────────────────────────────────────────────
RNG = np.random.default_rng(0)

N         = 20
MAX_DUR   = 77.0
P95       = 500.0

# ── synthetic matrices (all ndarray as they now come from problem_data) ───────
reward_mat = RNG.standard_normal((N, N)).astype(np.float64) * 200
np.fill_diagonal(reward_mat, -1e9)          # simulate BIG_M_PENALTY on diagonal
time_mat   = RNG.uniform(5.0, 60.0, (N, N)).astype(np.float64)
dist_mat   = RNG.uniform(100.0, 1000.0, (N, N)).astype(np.float64)
trucks_s   = RNG.uniform(0, 45, (N, 3)).astype(np.float32)
avail_p    = RNG.uniform(0, 1, (N, N)).astype(np.float32)

current_node = 5
start_node   = 0
visited_set  = {0, 3, 5}

TRUCKS_CLIP = 50.0  # from config

# ── Reference implementation of build_node_features (old loop version) ────────
def build_node_features_ref(current_node, start_node, visited_set,
                             reward_matrix_penalized, time_matrix,
                             distance_arr, num_nodes, max_duration,
                             trucks_stack=None, time_matrix_arr=None,
                             avail_prob_arr=None, reward_global_p95=1.0):
    from config import TRUCKS_CLIP as TC
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
            feats[j, 5] = min(trucks_val, TC) / TC
        if avail_prob_arr is not None:
            feats[j, 6] = float(avail_prob_arr[current_node, j])

    feats[:, 0] = np.clip(feats[:, 0], -1e4, 1e4)
    raw_rewards = feats[:, 0].copy()
    for col in (0, 2):
        col_min = feats[:, col].min()
        col_max = feats[:, col].max()
        if col_max > col_min:
            feats[:, col] = (feats[:, col] - col_min) / (col_max - col_min)
        else:
            feats[:, col] = 0.0
    feats[:, 1] = np.clip(feats[:, 1] / max(max_duration, 1e-6), 0.0, 1.0)
    _denom = max(reward_global_p95, 1e-6)
    feat7 = np.clip(raw_rewards / _denom, 0.0, 1.0)
    feat7[raw_rewards < 0] = 0.0
    feats[:, 7] = feat7
    return feats


# ── Reference implementation of _get_action_mask (old loop version) ──────────
def get_action_mask_ref(time_matrix, num_nodes, current_node, start_node,
                        visited_set, time_elapsed, max_duration,
                        current_lane_exists=None):
    mask = np.ones(num_nodes, dtype=np.int8)
    mask[current_node] = 0
    for v in visited_set:
        if v != start_node:
            mask[v] = 0
    for j in range(num_nodes):
        if mask[j] == 1 and j != start_node:
            t_to_j      = float(time_matrix[current_node][j])
            t_j_to_dep  = float(time_matrix[j][start_node])
            if time_elapsed + t_to_j + t_j_to_dep > max_duration:
                mask[j] = 0
    if current_lane_exists is not None and current_node != start_node:
        for j in range(num_nodes):
            if mask[j] == 1 and j != start_node:
                if current_lane_exists[j] == 0:
                    mask[j] = 0
    return mask


# ── Test 1: build_node_features ───────────────────────────────────────────────
print("Test 1: build_node_features vectorized vs reference loop ... ", end="", flush=True)
sys.path.insert(0, os.path.dirname(__file__))
from attention_encoder import build_node_features

ref = build_node_features_ref(
    current_node, start_node, visited_set,
    reward_mat, time_mat, dist_mat, N, MAX_DUR,
    trucks_stack=trucks_s,
    time_matrix_arr=time_mat,
    avail_prob_arr=avail_p,
    reward_global_p95=P95,
)
new = build_node_features(
    current_node, start_node, visited_set,
    reward_mat, time_mat, dist_mat, N, MAX_DUR,
    trucks_stack=trucks_s,
    time_matrix_arr=time_mat,
    avail_prob_arr=avail_p,
    reward_global_p95=P95,
)
assert new.shape == ref.shape, f"Shape mismatch: {new.shape} vs {ref.shape}"
if not np.allclose(new, ref, atol=1e-5):
    diff_cols = np.where(~np.isclose(new, ref, atol=1e-5))
    max_err = np.abs(new - ref).max()
    print(f"FAIL  max_err={max_err:.2e}  cols={np.unique(diff_cols[1])}")
    sys.exit(1)
print("OK")


# ── Test 2: build_node_features — no trucks / no avail ───────────────────────
print("Test 2: build_node_features without trucks/avail ... ", end="", flush=True)
ref2 = build_node_features_ref(current_node, start_node, visited_set,
                                reward_mat, time_mat, dist_mat, N, MAX_DUR)
new2 = build_node_features(current_node, start_node, visited_set,
                            reward_mat, time_mat, dist_mat, N, MAX_DUR)
assert np.allclose(new2, ref2, atol=1e-5), f"FAIL max_err={np.abs(new2-ref2).max():.2e}"
print("OK")


# ── Test 3: _get_action_mask — random lane availability ───────────────────────
print("Test 3: _get_action_mask vectorized vs reference loop ... ", end="", flush=True)
lane_exists = (RNG.uniform(0, 1, N) > 0.3).astype(np.int8)
time_elapsed = 20.0

ref3 = get_action_mask_ref(time_mat, N, current_node, start_node, visited_set,
                            time_elapsed, MAX_DUR, lane_exists)
# Use RoutingEnv internals directly
from routing_env import RoutingEnv
rate_stack  = np.zeros((3, N, N))
loads_stack = np.ones((3, N, N))
env = RoutingEnv(time_mat, rate_stack, loads_stack, dist_mat,
                 np.zeros((N, N)), avail_p, N, MAX_DUR)
env._current_node      = current_node
env._start_node        = start_node
env._visited_set       = set(visited_set)
env._time_elapsed      = time_elapsed
env._current_lane_exists = lane_exists
new3 = env._get_action_mask()

if not np.array_equal(new3, ref3):
    diff_idx = np.where(new3 != ref3)[0]
    print(f"FAIL  diff at indices {diff_idx}")
    print(f"  ref={ref3[diff_idx]}  new={new3[diff_idx]}")
    sys.exit(1)
print("OK")


# ── Test 4: _get_action_mask — no lane filter (current_node == start_node) ────
print("Test 4: _get_action_mask no lane filter (at depot) ... ", end="", flush=True)
env._current_node        = start_node
env._current_lane_exists = lane_exists
ref4 = get_action_mask_ref(time_mat, N, start_node, start_node, visited_set,
                            time_elapsed, MAX_DUR, lane_exists)
new4 = env._get_action_mask()
assert np.array_equal(new4, ref4), f"FAIL diff at {np.where(new4 != ref4)[0]}"
print("OK")


# ── Test 5: build_day_matrices returns ndarray ─────────────────────────────────
print("Test 5: build_day_matrices returns np.ndarray ... ", end="", flush=True)
from problem_data import build_day_matrices
rate_day  = RNG.uniform(0, 10, (N, N))
loads_day = RNG.uniform(0, 5, (N, N))
diesel_arr = RNG.uniform(3, 7, (N, N))
rm, rm_pen = build_day_matrices(rate_day, loads_day, dist_mat, diesel_arr)
assert isinstance(rm, np.ndarray),     f"reward_matrix should be ndarray, got {type(rm)}"
assert isinstance(rm_pen, np.ndarray), f"rm_penalized should be ndarray, got {type(rm_pen)}"
assert rm.dtype.kind == 'f',  f"Expected float dtype, got {rm.dtype}"
assert rm_pen[0, 0] == -1e9, f"Diagonal penalty wrong: {rm_pen[0,0]}"
print("OK")


# ── Test 6: _rm_pen_cache in RoutingEnv reuses without recompute ──────────────
print("Test 6: RoutingEnv._rm_pen_cache works correctly ... ", end="", flush=True)
env2 = RoutingEnv(time_mat, np.zeros((3, N, N)), np.ones((3, N, N)),
                  dist_mat, np.zeros((N, N)), avail_p, N, MAX_DUR)
env2._start_node  = 0
env2._start_day_idx = 0
# Two calls for the same day should return the same object (cache hit)
r1 = env2._get_rm_pen(0)
r2 = env2._get_rm_pen(0)
assert r1 is r2, "Cache miss: _get_rm_pen returned different objects for same day"
assert isinstance(r1, np.ndarray), f"Expected ndarray from cache, got {type(r1)}"
print("OK")


print("\nAll equivalence tests PASSED.")
