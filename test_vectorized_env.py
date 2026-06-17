"""
test_vectorized_env.py
======================
Regression test: VectorRoutingEnv must produce identical MDP trajectories
to the scalar RoutingEnv for the same (start_node, start_day_idx) pairs and
the same deterministic policy.

Deterministic policy: always pick the highest-index valid action from the mask.

Tests
-----
1. Same sequence of current_node stored per step.
2. Same sequence of masks (N,) int8 at each step.
3. Same reward per step.
4. Same done flag per step.
5. Same episode length (number of transitions).
6. Total buffer length equals sum of scalar episode lengths.
7. Edge case: at least one episode terminates exactly at step num_nodes-1;
   verify truncated_flags == False and last_values == 0.0 for that episode.
"""

import numpy as np
import pytest

from config import MAX_DURATION, BIG_M_PENALTY, REWARD_SCALE_FACTOR
from routing_env import RoutingEnv, VectorRoutingEnv
from problem_data import build_day_matrices, build_rm_pen_stack, draw_lane_availability


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic problem data
# ─────────────────────────────────────────────────────────────────────────────

NUM_NODES  = 6
NUM_DAYS   = 10
# Travel time = 1h for all arcs → an episode visiting all N nodes takes exactly
# N steps, last step (step N-1) returns to depot within MAX_DURATION_TEST.
MAX_DURATION_TEST = float(NUM_NODES) + 0.5   # 6.5h — allows full tour in N steps

RNG = np.random.default_rng(42)


def make_synthetic_data():
    N, D = NUM_NODES, NUM_DAYS

    # Uniform travel time = 1.0h for all arcs (diagonal irrelevant; env never uses it)
    time_matrix = np.ones((N, N), dtype=float)
    np.fill_diagonal(time_matrix, 0.0)

    # Rates and loads — small positive values so rewards are computable
    rate_stack  = RNG.uniform(1.0, 5.0, size=(D, N, N)).astype(float)
    loads_stack = np.ones((D, N, N), dtype=float) * 2.0   # all loads > 1 (so revenue > 0)
    distance_arr = np.ones((N, N), dtype=float) * 100.0
    np.fill_diagonal(distance_arr, 0.0)
    diesel_arr = np.ones((N, N), dtype=float) * 0.5

    # All lanes always available → stochastic draws are deterministic (Bernoulli(1))
    avail_prob_arr = np.ones((N, N), dtype=np.float32)

    return time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, avail_prob_arr


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic policy
# ─────────────────────────────────────────────────────────────────────────────

def greedy_action(mask: np.ndarray) -> int:
    """Pick the highest-index node that is valid (mask[j] == 1)."""
    valid = np.where(mask == 1)[0]
    assert len(valid) > 0, "Empty mask — no valid action"
    return int(valid[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Scalar rollout helper
# ─────────────────────────────────────────────────────────────────────────────

def run_scalar_episode(env: RoutingEnv, start_node: int, start_day_idx: int, num_nodes: int):
    """Run one episode with the deterministic policy. Returns list of transition dicts."""
    obs, info = env.reset(options={"start_node": start_node, "start_day_idx": start_day_idx})
    transitions = []
    for _ in range(num_nodes):
        mask         = info["action_mask"]
        current_node = env.current_node
        action       = greedy_action(mask)
        next_obs, reward, terminated, _, info = env.step(action)
        transitions.append({
            "current_node": current_node,
            "mask":         mask.copy(),
            "action":       action,
            "reward":       float(reward),
            "done":         bool(terminated),
        })
        if terminated:
            break
    return transitions


# ─────────────────────────────────────────────────────────────────────────────
# Main test
# ─────────────────────────────────────────────────────────────────────────────

def test_vectorized_equivalence():
    np.random.seed(0)

    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, avail_prob_arr = (
        make_synthetic_data()
    )
    N          = NUM_NODES
    num_days   = NUM_DAYS
    num_train_days = num_days

    rm_pen_stack = build_rm_pen_stack(
        rate_stack[:num_train_days], loads_stack[:num_train_days], distance_arr, diesel_arr
    )   # (num_train_days, N, N)

    # Four episodes: start_node 0-3, varied start days
    B             = 4
    start_nodes   = np.array([0, 1, 2, 3], dtype=np.int64)
    start_day_idxs = np.array([0, 2, 5, 7], dtype=np.int64)

    # ── Scalar reference ──────────────────────────────────────────────────────
    scalar_env = RoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        distance_arr=distance_arr,
        diesel_arr=diesel_arr,
        avail_prob_arr=avail_prob_arr,
        num_nodes=N,
        max_duration=MAX_DURATION_TEST,
    )

    scalar_episodes = []
    for b in range(B):
        ep = run_scalar_episode(
            scalar_env, int(start_nodes[b]), int(start_day_idxs[b]), N
        )
        scalar_episodes.append(ep)

    # ── Vectorized rollout ────────────────────────────────────────────────────
    vec_env = VectorRoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        distance_arr=distance_arr,
        diesel_arr=diesel_arr,
        avail_prob_arr=avail_prob_arr,
        rm_pen_stack=rm_pen_stack,
        num_nodes=N,
        max_duration=MAX_DURATION_TEST,
    )

    ep_bufs        = [[] for _ in range(B)]
    last_values    = np.zeros(B, dtype=np.float32)
    truncated_flags = np.zeros(B, dtype=bool)

    masks  = vec_env.reset(start_nodes, start_day_idxs)
    active = np.ones(B, dtype=bool)

    for step_i in range(N):   # defensive ceiling
        if not active.any():
            break

        cur_nodes = vec_env.current_node.copy()

        # Deterministic actions: highest valid index per episode
        actions = np.array([greedy_action(masks[b]) for b in range(B)], dtype=np.int64)

        rewards, terminated, next_masks = vec_env.step(actions)

        for b in np.where(active)[0]:
            ep_bufs[b].append({
                "current_node": int(cur_nodes[b]),
                "mask":         masks[b].copy(),
                "action":       int(actions[b]),
                "reward":       float(rewards[b]),
                "done":         bool(terminated[b]),
            })
            if terminated[b]:
                last_values[b] = 0.0

        active = active & ~terminated
        masks  = next_masks

    # Handle truncated (still active after step ceiling)
    for b in np.where(active)[0]:
        truncated_flags[b] = True
        # (no bootstrap value needed for this test — just mark truncated)

    # ── Assertions ────────────────────────────────────────────────────────────

    total_scalar_len = sum(len(ep) for ep in scalar_episodes)
    total_vec_len    = sum(len(ep_bufs[b]) for b in range(B))
    assert total_scalar_len == total_vec_len, (
        f"Buffer length mismatch: scalar={total_scalar_len}, vec={total_vec_len}"
    )

    for b in range(B):
        sc = scalar_episodes[b]
        vc = ep_bufs[b]
        assert len(sc) == len(vc), (
            f"Episode {b}: length mismatch scalar={len(sc)} vec={len(vc)}"
        )
        for t, (s, v) in enumerate(zip(sc, vc)):
            assert s["current_node"] == v["current_node"], (
                f"Episode {b} step {t}: current_node "
                f"scalar={s['current_node']} vec={v['current_node']}"
            )
            np.testing.assert_array_equal(
                s["mask"], v["mask"],
                err_msg=f"Episode {b} step {t}: mask mismatch",
            )
            np.testing.assert_allclose(
                s["reward"], v["reward"], rtol=1e-5, atol=1e-5,
                err_msg=f"Episode {b} step {t}: reward mismatch",
            )
            assert s["done"] == v["done"], (
                f"Episode {b} step {t}: done scalar={s['done']} vec={v['done']}"
            )

    # ── Edge case: episode 0 (start_node=0) should terminate at step N-1 ─────
    # With time_matrix[i,j]=1 and max_duration=6.5, visiting all N=6 nodes
    # takes exactly N steps (steps 0..N-1), terminating at step N-1.
    ep0_len = len(ep_bufs[0])
    assert ep0_len == N, (
        f"Edge case failed: episode 0 expected length {N} (terminates at step N-1), "
        f"got {ep0_len}"
    )
    last_step_0 = ep_bufs[0][-1]
    assert last_step_0["done"] is True, "Edge case: last step of episode 0 should be done"
    assert not truncated_flags[0], (
        "Edge case: episode 0 terminated naturally at step N-1 — must NOT be truncated"
    )
    assert last_values[0] == 0.0, (
        f"Edge case: last_value for terminated episode must be 0.0, got {last_values[0]}"
    )

    print(f"\nAll assertions passed.")
    print(f"  Scalar buffer: {total_scalar_len} transitions")
    print(f"  Vectorized buffer: {total_vec_len} transitions")
    for b in range(B):
        print(
            f"  Episode {b} (start_node={start_nodes[b]}, day={start_day_idxs[b]}): "
            f"len={len(ep_bufs[b])} truncated={truncated_flags[b]}"
        )


if __name__ == "__main__":
    test_vectorized_equivalence()
    print("test_vectorized_env.py: PASSED")
