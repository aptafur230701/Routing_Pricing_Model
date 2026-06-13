"""
test_mip_dynamic.py
===================
Validation tests for solve_mip_dynamic.

Test 1 — Canonical evaluation consistency: reported reward matches simulate_route_reward.
Test 2 — Structural validity: route structure, duration, length.
Test 3 — Dominance: MIP Reward >= reward of every deterministic heuristic (per instance).
Test 4 — Bucket consistency: MIP obj decomposed arc-by-arc matches canonical simulation.
Test 5 — DP verification (slow): Held-Karp style DP gives same optimal as MIP.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest

from problem_data import load_matrices, build_day_matrices
from Solvers import (
    solve_mip_dynamic,
    solve_HGA_LNS_metaheuristic,
    solve_heuristic_rolling_horizon,
    solve_heuristic_rolling_horizon_lookahead,
    simulate_route_reward,
    _day_index,
    DAYS_PER_PERIOD,
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
    return (time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr)


# Combinaciones de (start_node, day_idx) usadas en todos los tests.
COMBOS = [(s, d) for s in range(5) for d in range(3)]


# ── Verificador por DP (Held–Karp estilo) ────────────────────────────────────

def _dp_optimal(start_node, time_m, rate_eval, loads_eval,
                distance_arr, diesel_arr, max_d, num_n, start_day_idx):
    """
    Held-Karp sobre (frozenset visitados, nodo actual), manteniendo la frontera
    de Pareto sobre (tiempo acumulado, reward acumulado).  Retorna el mejor
    reward del ciclo óptimo determinista, equivalente al que calcula
    simulate_route_reward(..., avail_prob_arr=None).
    """
    max_day = rate_eval.shape[0] - 1

    # Caché de matrices de reward numpy (penalizadas) por día efectivo.
    _rm_cache = {}
    def _get_R(t_elapsed):
        d_eff = _day_index(start_day_idx, t_elapsed, max_day)
        if d_eff not in _rm_cache:
            _, rm_pen = build_day_matrices(
                rate_eval[d_eff], loads_eval[d_eff], distance_arr, diesel_arr
            )
            _rm_cache[d_eff] = np.array(rm_pen, dtype=float)
        return _rm_cache[d_eff]

    # Estado: (frozenset de intermedios visitados, nodo actual)
    # Valor: lista de (tiempo_acumulado, reward_acumulado) — frontera de Pareto
    # (menor tiempo para igual o mayor reward).
    init_state = (frozenset(), start_node)
    # {estado: list of (time, reward)}
    frontier = {init_state: [(0.0, 0.0)]}

    best_reward = -np.inf

    # BFS / DP sobre los pasos k = 0..num_n-1
    for _ in range(num_n):
        new_frontier = {}
        for (vis, cur), pareto in frontier.items():
            for nxt in range(num_n):
                if nxt == cur:
                    continue
                if nxt != start_node and nxt in vis:
                    continue
                # Si queremos salir al depósito desde un intermedio, solo en cierre.
                # Aquí permitimos el retorno en cualquier paso (como la restricción 4 del MIP).
                for (t, r) in pareto:
                    step_t = float(time_m[cur, nxt])
                    new_t = t + step_t
                    if new_t > max_d + 1e-9:
                        continue
                    if nxt != start_node:
                        ret_t = float(time_m[nxt, start_node])
                        if new_t + ret_t > max_d + 1e-9:
                            continue
                    arc_r = float(_get_R(t)[cur, nxt])
                    new_r = r + arc_r
                    if nxt == start_node:
                        # Ciclo cerrado
                        if new_t <= max_d + 1e-9:
                            best_reward = max(best_reward, new_r)
                    else:
                        new_vis = vis | {nxt}
                        key = (new_vis, nxt)
                        pts = new_frontier.setdefault(key, [])
                        # Insertar en frontera de Pareto (menor t es mejor para exploración futura)
                        dominated = False
                        to_remove = []
                        for idx, (pt, pr) in enumerate(pts):
                            if pt <= new_t and pr >= new_r:
                                dominated = True
                                break
                            if pt >= new_t and pr <= new_r:
                                to_remove.append(idx)
                        if not dominated:
                            for idx in reversed(to_remove):
                                pts.pop(idx)
                            pts.append((new_t, new_r))
        # Fusionar con frontier existente para permitir diferentes longitudes de ruta
        for key, pts in new_frontier.items():
            if key not in frontier:
                frontier[key] = pts
            else:
                # Merge Pareto
                combined = frontier[key] + pts
                merged = []
                combined.sort(key=lambda p: p[0])
                best_r = -np.inf
                for (pt, pr) in combined:
                    if pr > best_r:
                        merged.append((pt, pr))
                        best_r = pr
                frontier[key] = merged

    return best_reward


# ── Test 1 — Consistencia canónica ───────────────────────────────────────────

def test_canonical_consistency(problem_data):
    """Reward reportado == simulate_route_reward sobre la misma ruta."""
    tm, rate, loads, dist, diesel = problem_data

    for (start_node, day_idx) in COMBOS:
        result = solve_mip_dynamic(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx,
            time_limit_s=120,
        )
        if result.route is None:
            continue

        canon_r, _ = simulate_route_reward(
            result.route, start_node, day_idx,
            tm, rate, loads, dist, diesel,
            avail_prob_arr=None,
        )
        assert abs(result.reward - canon_r) < 1e-6, (
            f"start={start_node} day={day_idx}: MIP reported {result.reward:.6f} "
            f"but simulate_route_reward={canon_r:.6f} (diff={abs(result.reward - canon_r):.2e})"
        )


# ── Test 2 — Validez estructural ─────────────────────────────────────────────

def test_structural_validity(problem_data):
    """Ruta empieza/termina en start_node, intermedios únicos, duración ≤ MAX_DURATION."""
    tm, rate, loads, dist, diesel = problem_data

    for (start_node, day_idx) in COMBOS:
        result = solve_mip_dynamic(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx,
            time_limit_s=120,
        )
        if result.route is None:
            continue

        route = result.route
        assert route[0] == start_node,  f"start={start_node}: route doesn't start at depot"
        assert route[-1] == start_node, f"start={start_node}: route doesn't end at depot"

        intermediates = route[1:-1]
        assert len(intermediates) == len(set(intermediates)), (
            f"start={start_node} day={day_idx}: duplicate intermediate nodes in {route}"
        )

        duration = sum(float(tm[route[p], route[p + 1]]) for p in range(len(route) - 1))
        assert duration <= MAX_DURATION + 1e-6, (
            f"start={start_node} day={day_idx}: duration {duration:.4f} > MAX_DURATION {MAX_DURATION}"
        )

        assert len(route) <= NUM_NODES + 1, (
            f"start={start_node} day={day_idx}: route length {len(route)} > {NUM_NODES + 1}"
        )


# ── Test 3 — Dominancia (test crítico) ───────────────────────────────────────

def test_dominance(problem_data):
    """MIP Reward >= reward de cada heurístico determinista, instancia por instancia."""
    tm, rate, loads, dist, diesel = problem_data

    for (start_node, day_idx) in COMBOS:
        mip_result = solve_mip_dynamic(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx,
            time_limit_s=120,
        )
        if mip_result.status != 'Optimal' or mip_result.route is None:
            continue  # sin garantía de cota; no testear dominancia

        mip_r = mip_result.reward

        # HGA-LNS
        hga = solve_HGA_LNS_metaheuristic(
            start_node, tm, MAX_DURATION, NUM_NODES,
            rate_stack=rate, loads_stack=loads,
            distance_arr=dist, diesel_arr=diesel,
            start_day_idx=day_idx, seed=SEED,
        )
        if hga.status == 'Optimal' and hga.route is not None:
            assert mip_r >= hga.reward - 1e-6, (
                f"start={start_node} day={day_idx}: MIP {mip_r:.4f} < HGA-LNS {hga.reward:.4f}"
            )

        # RH-Greedy
        rh = solve_heuristic_rolling_horizon(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, start_day_idx=day_idx,
        )
        if rh.is_valid and rh.route is not None:
            assert mip_r >= rh.reward - 1e-6, (
                f"start={start_node} day={day_idx}: MIP {mip_r:.4f} < RH-Greedy {rh.reward:.4f}"
            )

        # RH-Lookahead
        rhl = solve_heuristic_rolling_horizon_lookahead(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, start_day_idx=day_idx, lookahead=3,
        )
        if rhl.is_valid and rhl.route is not None:
            assert mip_r >= rhl.reward - 1e-6, (
                f"start={start_node} day={day_idx}: MIP {mip_r:.4f} < RH-Lookahead {rhl.reward:.4f}"
            )


# ── Test 4 — Consistencia de buckets ─────────────────────────────────────────

def test_bucket_consistency(problem_data):
    """El objetivo MIP descompuesto arco a arco coincide con simulate_route_reward."""
    tm, rate, loads, dist, diesel = problem_data
    max_day = rate.shape[0] - 1

    _rm_cache = {}
    def _get_R_np(start_day_idx, t_elapsed):
        d_eff = _day_index(start_day_idx, t_elapsed, max_day)
        key = (start_day_idx, d_eff)
        if key not in _rm_cache:
            _, rm_pen = build_day_matrices(
                rate[d_eff], loads[d_eff], dist, diesel
            )
            _rm_cache[key] = np.array(rm_pen, dtype=float)
        return _rm_cache[key]

    for (start_node, day_idx) in COMBOS:
        result = solve_mip_dynamic(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx,
            time_limit_s=120,
        )
        if result.route is None:
            continue

        route = result.route
        t_acc = 0.0
        arc_sum = 0.0
        for p in range(len(route) - 1):
            i, j = route[p], route[p + 1]
            arc_r = float(_get_R_np(day_idx, t_acc)[i, j])
            arc_sum += arc_r
            t_acc += float(tm[i, j])

        canon_r, _ = simulate_route_reward(
            route, start_node, day_idx,
            tm, rate, loads, dist, diesel,
            avail_prob_arr=None,
        )
        assert abs(arc_sum - canon_r) < 1e-6, (
            f"start={start_node} day={day_idx}: arc-by-arc sum {arc_sum:.6f} "
            f"!= canonical {canon_r:.6f}"
        )


# ── Test 5 — Verificación cruzada por DP (lento) ─────────────────────────────

@pytest.mark.slow
def test_dp_verification(problem_data):
    """DP exacto (Held–Karp) da el mismo óptimo que el MIP en todas las instancias."""
    tm, rate, loads, dist, diesel = problem_data

    for (start_node, day_idx) in COMBOS:
        mip_result = solve_mip_dynamic(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx,
            time_limit_s=120,
        )
        if mip_result.status != 'Optimal' or mip_result.route is None:
            continue

        dp_opt = _dp_optimal(
            start_node, tm, rate, loads, dist, diesel,
            MAX_DURATION, NUM_NODES, day_idx,
        )
        assert abs(mip_result.reward - dp_opt) < 1e-6, (
            f"start={start_node} day={day_idx}: MIP={mip_result.reward:.6f} "
            f"DP={dp_opt:.6f} diff={abs(mip_result.reward - dp_opt):.2e}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
