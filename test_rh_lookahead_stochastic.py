"""
test_rh_lookahead_stochastic.py
================================
Verifica dos propiedades de solve_heuristic_rolling_horizon_lookahead_stochastic:

1. Equivalencia con lookahead=1: el reward final es idéntico al de
   solve_heuristic_rolling_horizon_stochastic para los mismos inputs.

2. Cota superior MIP-Exact (mundo determinista): con avail_prob_arr=None el
   reward reportado nunca supera al de solve_mip_exact, consistente con el
   invariante verificado por test_mip_exact_bound.py.

Ejecutar:
    python test_rh_lookahead_stochastic.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAX_DURATION, SEED
from problem_data import load_matrices
from Solvers import (
    solve_mip_exact,
    solve_heuristic_rolling_horizon_stochastic,
    solve_heuristic_rolling_horizon_lookahead_stochastic,
)

NUM_NODES   = 10
TOLERANCE   = 1.0
N_DAYS_TEST = 5


# ── Test 1: lookahead=1 equivale a solve_heuristic_rolling_horizon_stochastic ──

def test_lookahead1_equals_stochastic(time_matrix_np, rate_stack, loads_stack,
                                      distance_arr, diesel_arr, avail_prob_arr):
    print("\n=== Test 1: lookahead=1 == RH-Greedy Real ===")
    rng = np.random.default_rng(SEED)
    max_day = rate_stack.shape[0] - 1
    day_indices = rng.integers(0, max_day + 1, size=N_DAYS_TEST).tolist()

    fails = 0
    for start in range(NUM_NODES):
        for day_idx in day_indices:
            case_id = f"start={start} day={day_idx}"

            _, _, stoch_reward, _, stoch_valid = solve_heuristic_rolling_horizon_stochastic(
                start, time_matrix_np, rate_stack, loads_stack,
                distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
                start_day_idx=day_idx, avail_prob_arr=avail_prob_arr,
            )

            _, _, lkah_reward, _, lkah_valid = \
                solve_heuristic_rolling_horizon_lookahead_stochastic(
                    start, time_matrix_np, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
                    start_day_idx=day_idx, avail_prob_arr=avail_prob_arr,
                    lookahead=1,
                )

            if stoch_valid != lkah_valid or (
                stoch_valid and abs(stoch_reward - lkah_reward) > TOLERANCE
            ):
                print(
                    f"  [FAIL] {case_id}: stoch={stoch_reward:.2f} valid={stoch_valid} "
                    f"!= lookahead1={lkah_reward:.2f} valid={lkah_valid}"
                )
                fails += 1
            else:
                print(f"  OK  {case_id}: stoch={stoch_reward:.1f} == lookahead1={lkah_reward:.1f}")

    print(f"\nTest 1 completado: {fails} fallos de {NUM_NODES * N_DAYS_TEST} casos")
    return fails == 0


# ── Test 2: MIP-Exact >= lookahead_stochastic en mundo determinista ────────────

def test_mip_upper_bound_det(time_matrix_np, rate_stack, loads_stack,
                              distance_arr, diesel_arr):
    print("\n=== Test 2: MIP-Exact >= RH-Lookahead Real (det world) ===")
    rng = np.random.default_rng(SEED)
    max_day = rate_stack.shape[0] - 1
    day_indices = rng.integers(0, max_day + 1, size=N_DAYS_TEST).tolist()

    fails = 0
    for start in range(NUM_NODES):
        for day_idx in day_indices:
            case_id = f"start={start} day={day_idx}"

            mip_status, mip_route, mip_reward, _ = solve_mip_exact(
                start_node     = start,
                time_matrix_np = time_matrix_np,
                rate_stack     = rate_stack,
                loads_stack    = loads_stack,
                distance_arr   = distance_arr,
                diesel_arr     = diesel_arr,
                max_d          = MAX_DURATION,
                num_n          = NUM_NODES,
                start_day_idx  = day_idx,
            )
            mip_valid = (mip_route is not None and len(mip_route) > 1
                         and mip_route[0] == mip_route[-1])

            _, _, lkah_reward, _, lkah_valid = \
                solve_heuristic_rolling_horizon_lookahead_stochastic(
                    start, time_matrix_np, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
                    start_day_idx=day_idx, avail_prob_arr=None,  # mundo determinista
                    lookahead=3,
                )

            if not mip_valid or not lkah_valid:
                print(f"  [SKIP] {case_id} — mip_valid={mip_valid} lkah_valid={lkah_valid}")
                continue

            if lkah_reward > mip_reward + TOLERANCE:
                print(
                    f"  [FAIL] {case_id}: RH-Lookahead Real det={lkah_reward:.2f} "
                    f"> MIP-Exact={mip_reward:.2f} (gap={lkah_reward - mip_reward:.2f})"
                )
                fails += 1
            else:
                print(f"  OK  {case_id}: MIP-Exact={mip_reward:.1f} >= "
                      f"RH-Lookahead Real={lkah_reward:.1f} "
                      f"(gap={mip_reward - lkah_reward:.1f})")

    print(f"\nTest 2 completado: {fails} fallos de {NUM_NODES * N_DAYS_TEST} casos")
    return fails == 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  test_rh_lookahead_stochastic")
    print("=" * 60)

    print("\nCargando matrices...")
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr, _) = load_matrices(NUM_NODES)
    time_matrix_np = np.array(time_matrix, dtype=float)
    print(f"Nodos: {NUM_NODES} | días: {rate_stack.shape[0]} | MAX_DURATION: {MAX_DURATION}h")

    results = []
    results.append(
        test_lookahead1_equals_stochastic(
            time_matrix_np, rate_stack, loads_stack,
            distance_arr, diesel_arr, avail_prob_arr,
        )
    )
    results.append(
        test_mip_upper_bound_det(
            time_matrix_np, rate_stack, loads_stack,
            distance_arr, diesel_arr,
        )
    )

    print("\n" + "=" * 60)
    if all(results):
        print("  TODOS LOS TESTS PASARON")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f"  {n_fail} TEST(S) FALLARON")
        sys.exit(1)


if __name__ == "__main__":
    main()
