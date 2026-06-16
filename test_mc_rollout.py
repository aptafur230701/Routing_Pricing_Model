"""
test_mc_rollout.py
==================
Validation tests for solve_mc_rollout_stochastic.

Test 1 — Policy improvement: MC-Rollout reward >= RH-Greedy reward (in expectation).
Test 2 — OBSOLETO: Convergence with n_simulations (eliminado). draw_lane_availability es
         determinista, por lo que todas las trayectorias por candidato eran idénticas;
         n_simulations ya no tiene efecto real (default=1).
Test 3 — Canonical evaluation consistency: reported reward matches simulate_route_reward.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from problem_data import load_matrices
from Solvers import (
    solve_mc_rollout_stochastic,
    solve_heuristic_rolling_horizon_stochastic,
    simulate_route_reward,
)
from config import SEED, TRAIN_DAYS, MAX_DURATION


NUM_NODES = 10


@pytest.fixture(scope="module")
def problem_data():
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr, _) = load_matrices(NUM_NODES)
    rate_eval  = rate_stack[TRAIN_DAYS:]
    loads_eval = loads_stack[TRAIN_DAYS:]
    time_matrix_np = np.array(time_matrix, dtype=float)
    return (time_matrix_np, rate_eval, loads_eval,
            distance_arr, diesel_arr, avail_prob_arr)


def _run_pair(problem_data, start_node, day_idx, n_sim=30):
    """Return (mc_reward, rh_reward) for one (start_node, day_idx) combo."""
    tm, rate, loads, dist, diesel, avail = problem_data

    _, _, mc_r, _, mc_valid = solve_mc_rollout_stochastic(
        start_node, tm, rate, loads, dist, diesel,
        MAX_DURATION, NUM_NODES, day_idx, avail,
        n_simulations=n_sim,
    )

    _, _, rh_r, _, rh_valid = solve_heuristic_rolling_horizon_stochastic(
        start_node, tm, rate, loads, dist, diesel,
        MAX_DURATION, NUM_NODES, day_idx, avail,
    )

    return (mc_r if mc_valid else -np.inf,
            rh_r if rh_valid else -np.inf)


# ── Test 1 — Policy improvement ───────────────────────────────────────────────

def test_policy_improvement(problem_data):
    """MC-Rollout reward >= RH-Greedy reward in at least 80% of cases."""
    combos = [(s, d) for s in range(5) for d in range(5)][:10]
    wins = 0
    losses = 0

    for (start_node, day_idx) in combos:
        mc_r, rh_r = _run_pair(problem_data, start_node, day_idx)
        if mc_r >= rh_r - 1e-6:
            wins += 1
        else:
            losses += 1
            print(f"  [warn] start={start_node} day={day_idx}: "
                  f"MC-Rollout {mc_r:.1f} < RH-Greedy {rh_r:.1f}")

    win_rate = wins / len(combos)
    print(f"\n  Policy improvement: {wins}/{len(combos)} = {win_rate:.0%}")
    assert win_rate >= 0.80, (
        f"MC-Rollout underperformed RH-Greedy in {losses}/{len(combos)} cases "
        f"(threshold 20%). This may indicate a bug."
    )


# ── Test 2 — OBSOLETO (eliminado) ─────────────────────────────────────────────
# test_convergence_with_simulations fue eliminado. draw_lane_availability es
# determinista dado (start_day_idx, node, arrival_day), así que todas las
# trayectorias por candidato producían el mismo valor. Promediar n copias
# idénticas nunca redujo varianza, y n_simulations=1 es ahora el default.

# ── Test 3 — Canonical evaluation consistency ─────────────────────────────────

def test_canonical_evaluation_consistency(problem_data):
    """Reported reward equals simulate_route_reward on the same route."""
    tm, rate, loads, dist, diesel, avail = problem_data

    for start_node in range(NUM_NODES):
        day_idx = start_node % rate.shape[0]
        _, route, mc_reward, _, mc_valid = solve_mc_rollout_stochastic(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx, avail,
            n_simulations=5,
        )
        if not mc_valid or route is None:
            continue

        canon_reward, _ = simulate_route_reward(
            route, start_node, day_idx,
            tm, rate, loads, dist, diesel,
            avail_prob_arr=avail,
        )

        assert abs(mc_reward - canon_reward) < 1e-6, (
            f"start={start_node} day={day_idx}: "
            f"MC-Rollout reported {mc_reward:.6f} but simulate_route_reward "
            f"returned {canon_reward:.6f} (diff={abs(mc_reward - canon_reward):.2e})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
