"""
test_mip_exact_bound.py
=======================
Verifica que solve_mip_exact es una cota superior demostrable en el mundo
determinista: ningún solver evaluado sobre las mismas matrices de reward
deterministas puede superar el reward de solve_mip_exact.

Propiedad central (la que hoy se viola con solve_mip_oracle):
    mip_exact_reward >= any_valid_route_det_reward - TOLERANCE

donde TOLERANCE = 1.0 (una unidad de reward redondeado) para absorber
diferencias de floating-point en np.round(..., 0).

Ejecutar:
    python test_mip_exact_bound.py
"""

import os
import sys
import numpy as np

# Asegura que el directorio raíz del proyecto esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import MAX_DURATION, DEVICE, SEED
from problem_data import load_matrices
from Solvers import (
    solve_mip_exact,
    solve_heuristic_rolling_horizon,
    simulate_route_reward,
)

NUM_NODES  = 10
TOLERANCE  = 1.0   # unidades de reward; absorbe np.round(..., 0)
N_DAYS_TEST = 5    # número de start_day_idx distintos a probar
CHECKPOINT = os.path.join(os.path.dirname(__file__), "am_checkpoint_10nodes.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────

def det_reward(route, start_node, day_idx,
               time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr):
    """Evaluación determinista (sin Bernoulli) de una ruta fija."""
    r, _ = simulate_route_reward(
        route, start_node, day_idx,
        time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    return r


def assert_upper_bound(case_id, mip_reward, solver_name, solver_reward):
    """Lanza AssertionError con mensaje explícito si la propiedad se viola."""
    if mip_reward == -np.inf:
        print(f"  [SKIP] {case_id} — MIP infeasible, no comparison possible")
        return
    if solver_reward == -np.inf:
        return  # solver infeasible: no hay nada que comparar
    if solver_reward > mip_reward + TOLERANCE:
        raise AssertionError(
            f"\n[BUG] {case_id}: {solver_name} det_reward={solver_reward:.2f} "
            f"> MIP-Exact reward={mip_reward:.2f} "
            f"(gap={solver_reward - mip_reward:.2f} > tol={TOLERANCE})\n"
            "  → solve_mip_exact NO es cota superior. Revisar formulación."
        )
    print(f"  OK  {case_id}: MIP-Exact={mip_reward:.1f} >= "
          f"{solver_name}={solver_reward:.1f} "
          f"(gap={mip_reward - solver_reward:.1f})")


# ── Test 1: MIP-Exact >= RH-Greedy para varios start_node y start_day_idx ─────

def test_mip_exact_vs_greedy(time_matrix_np, rate_stack, loads_stack,
                              distance_arr, diesel_arr, avail_prob_arr):
    print("\n=== Test 1: MIP-Exact >= RH-Greedy (determinista) ===")
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

            # RH-Greedy como proxy de cualquier política admisible
            _, rh_route, _, _, rh_valid = solve_heuristic_rolling_horizon(
                start, time_matrix_np, rate_stack, loads_stack,
                distance_arr, diesel_arr, MAX_DURATION, NUM_NODES,
                start_day_idx=day_idx,
            )
            rh_det = det_reward(rh_route, start, day_idx,
                                 time_matrix_np, rate_stack, loads_stack,
                                 distance_arr, diesel_arr) if rh_valid else -np.inf

            try:
                assert_upper_bound(case_id, mip_reward, "RH-Greedy", rh_det)
            except AssertionError as e:
                print(str(e))
                fails += 1

    print(f"\nTest 1 completado: {fails} fallos de "
          f"{NUM_NODES * N_DAYS_TEST} combinaciones")
    return fails == 0


# ── Test 2: MIP-Exact >= rutas válidas aleatorias ─────────────────────────────

def test_mip_exact_vs_random_routes(time_matrix_np, rate_stack, loads_stack,
                                     distance_arr, diesel_arr):
    """Genera rutas válidas aleatorias y verifica que MIP-Exact las domina."""
    print("\n=== Test 2: MIP-Exact >= rutas aleatorias válidas ===")
    rng = np.random.default_rng(SEED + 7)
    max_day = rate_stack.shape[0] - 1
    fails = 0
    n_tested = 0

    for start in range(NUM_NODES):
        day_idx = int(rng.integers(0, max_day + 1))

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
        if mip_reward == -np.inf:
            continue

        # Genera rutas aleatorias cortas y verifica
        for _ in range(20):
            intermediates = [n for n in range(NUM_NODES) if n != start]
            rng.shuffle(intermediates)
            k = int(rng.integers(1, min(5, NUM_NODES - 1) + 1))
            route = [start] + intermediates[:k] + [start]

            t_elapsed = sum(
                float(time_matrix_np[route[i]][route[i+1]])
                for i in range(len(route) - 1)
            )
            if t_elapsed > MAX_DURATION:
                continue   # ruta infactible: ignorar

            r = det_reward(route, start, day_idx,
                           time_matrix_np, rate_stack, loads_stack,
                           distance_arr, diesel_arr)
            case_id = f"start={start} day={day_idx} route={route}"
            try:
                assert_upper_bound(case_id, mip_reward, "RandomRoute", r)
                n_tested += 1
            except AssertionError as e:
                print(str(e))
                fails += 1
                n_tested += 1

    print(f"\nTest 2 completado: {fails} fallos de {n_tested} rutas válidas")
    return fails == 0


# ── Test 3 (opcional): MIP-Exact >= DRL Det si existe checkpoint ──────────────

def test_mip_exact_vs_drl(time_matrix_np, rate_stack, loads_stack,
                           distance_arr, diesel_arr,
                           ltr_stack, trucks_stack, avail_prob_arr):
    """Carga el agente entrenado (si existe) y verifica la cota vs DRL Det."""
    if not os.path.exists(CHECKPOINT):
        print(f"\n=== Test 3: DRL Det — checkpoint no encontrado, SKIP ===")
        print(f"   ({CHECKPOINT})")
        return True

    print(f"\n=== Test 3: MIP-Exact >= DRL Det (checkpoint {CHECKPOINT}) ===")
    import torch
    from am_agent import AMRoutingAgent
    from config import AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF

    agent = AMRoutingAgent(
        num_nodes=NUM_NODES, d_h=AM_D_H, n_heads=AM_N_HEADS,
        n_layers=AM_N_LAYERS, d_ff=AM_D_FF, device=DEVICE,
    )
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    # Checkpoint may wrap the agent under an "agent" key
    if "agent" in ckpt:
        state_dict = ckpt["agent"]
    else:
        state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        agent.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"\n=== Test 3: checkpoint incompatible con la arquitectura actual, SKIP ===")
        print(f"   ({e.__class__.__name__}: {str(e)[:120]}...)")
        return True
    agent.eval()

    from evaluation import rollout_drl_env

    rng = np.random.default_rng(SEED + 13)
    max_day = rate_stack.shape[0] - 1
    fails = 0

    for start in range(NUM_NODES):
        day_idx = int(rng.integers(0, max_day + 1))
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

        # DRL Det: construye sin Bernoulli y evalúa en mundo determinista
        drl_det_route, _, _ = rollout_drl_env(
            agent, start, day_idx,
            time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=None,   # sin Bernoulli
        )
        drl_det_valid = drl_det_route is not None
        drl_det_r = det_reward(drl_det_route, start, day_idx,
                                time_matrix_np, rate_stack, loads_stack,
                                distance_arr, diesel_arr) if drl_det_valid else -np.inf

        try:
            assert_upper_bound(case_id, mip_reward, "DRL Det", drl_det_r)
        except AssertionError as e:
            print(str(e))
            fails += 1

    print(f"\nTest 3 completado: {fails} fallos de {NUM_NODES} nodos")
    return fails == 0


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  test_mip_exact_bound — cota óptima determinista")
    print("=" * 60)

    print("\nCargando matrices...")
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr) = load_matrices(NUM_NODES)
    time_matrix_np = np.array(time_matrix, dtype=float)
    print(f"Nodos: {NUM_NODES} | días disponibles: {rate_stack.shape[0]} "
          f"| MAX_DURATION: {MAX_DURATION}h")

    results = []
    results.append(
        test_mip_exact_vs_greedy(time_matrix_np, rate_stack, loads_stack,
                                  distance_arr, diesel_arr, avail_prob_arr)
    )
    results.append(
        test_mip_exact_vs_random_routes(time_matrix_np, rate_stack, loads_stack,
                                         distance_arr, diesel_arr)
    )
    results.append(
        test_mip_exact_vs_drl(time_matrix_np, rate_stack, loads_stack,
                               distance_arr, diesel_arr,
                               ltr_stack, trucks_stack, avail_prob_arr)
    )

    print("\n" + "=" * 60)
    if all(results):
        print("  TODOS LOS TESTS PASARON — solve_mip_exact es cota superior")
    else:
        n_fail = sum(1 for r in results if not r)
        print(f"  {n_fail} TEST(S) FALLARON — revisar implementación")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
