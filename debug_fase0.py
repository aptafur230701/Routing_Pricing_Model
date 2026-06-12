"""
debug_fase0.py
==============
Script de diagnóstico standalone para el bug:

  start_node=15, eval_day=23 (abs_day=113):
  DRL Det  reward 3787.0  >  MIP-Exact reward 2885.0
  (viola la propiedad de cota superior del MIP)

Hipótesis evaluadas
-------------------
  H1: DRL Det y MIP-Exact reciben day_idx distintos
  H2: simulate_route_reward y solve_mip_exact usan distintas reward matrices
  H3: DRL Det fue evaluado con Bernoulli activo (avail_prob_arr != None)
  H4: floating point / np.round inconsistency entre los dos paths

Uso:
  python debug_fase0.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SEED, TRAIN_DAYS, MAX_DURATION
from problem_data import load_matrices, build_day_matrices
from Solvers import simulate_route_reward, solve_mip_exact

NUM_NODES = 20
START_NODE = 15
REPORTED_EVAL_DAY = 23      # log: "eval day 23"
REPORTED_ABS_DAY  = 113     # log: "abs day 113" = TRAIN_DAYS + 23

DRL_DET_ROUTE  = [15, 11, 13, 1, 15]
MIP_EXACT_ROUTE = [15, 12, 16, 1, 15]

DRL_DET_REPORTED_REWARD  = 3787.0
MIP_EXACT_REPORTED_REWARD = 2885.0

W = 72


def sep(title=""):
    if title:
        dash = max(0, W - 4 - len(title))
        print(f"\n{'─'*2} {title} {'─'*dash}")
    else:
        print("─" * W)


def main():
    print("=" * W)
    print("DEBUG FASE 0 — start_node=15, eval_day=23, abs_day=113")
    print(f"  Bug: DRL Det {DRL_DET_REPORTED_REWARD} > MIP-Exact {MIP_EXACT_REPORTED_REWARD}")
    print("=" * W)

    # ── 1. Cargar datos ──────────────────────────────────────────────────────
    sep("1. Carga de datos con load_matrices(20)")
    (time_matrix, rate_stack, loads_stack,
     distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr, _) = load_matrices(NUM_NODES)

    print(f"  rate_stack.shape (full): {rate_stack.shape}")
    print(f"  TRAIN_DAYS = {TRAIN_DAYS}")

    rate_eval   = rate_stack[TRAIN_DAYS:]
    loads_eval  = loads_stack[TRAIN_DAYS:]
    num_eval_days = rate_eval.shape[0]
    max_day = num_eval_days - 1
    print(f"  rate_eval.shape  (eval): {rate_eval.shape}  (índices 0..{max_day})")

    time_matrix_np = np.array(time_matrix, dtype=float)

    # ── 2. Reproducir day_idx de run_solver_comparison ──────────────────────
    sep("2. Reproducir asignación de días (mismo RNG que run_solver_comparison)")
    rng = np.random.default_rng(SEED)
    day_indices = rng.integers(0, num_eval_days, size=NUM_NODES)
    print(f"  day_indices[0..{NUM_NODES-1}] = {day_indices.tolist()}")

    day_idx_reproduced = int(day_indices[START_NODE])
    abs_day_reproduced = TRAIN_DAYS + day_idx_reproduced
    print(f"\n  Para start_node={START_NODE}:")
    print(f"    day_idx reproducido  = {day_idx_reproduced}  "
          f"{'✓ coincide con el log' if day_idx_reproduced == REPORTED_EVAL_DAY else '✗ DIFIERE del log'}")
    print(f"    abs_day reproducido  = {abs_day_reproduced}  "
          f"{'✓ coincide con el log' if abs_day_reproduced == REPORTED_ABS_DAY else '✗ DIFIERE del log'}")

    # Elegir el day_idx a usar para las pruebas
    if day_idx_reproduced == REPORTED_EVAL_DAY:
        day_idx = day_idx_reproduced
        print(f"  → Usando day_idx={day_idx} (reproducido del log).")
    else:
        day_idx = REPORTED_EVAL_DAY
        print(f"  [ADVERTENCIA] RNG no reproduce el log. "
              f"Forzando day_idx={day_idx} como indica el log.")

    # Verificar que day_idx es válido para el eval set
    if day_idx > max_day:
        print(f"  [ERROR] day_idx={day_idx} > max_day={max_day}. "
              f"El eval set tiene solo {num_eval_days} días.")
        sys.exit(1)

    # ── H1: ¿Ambos solvers reciben el mismo day_idx? ─────────────────────────
    sep("H1 — ¿DRL Det y MIP-Exact reciben el mismo day_idx?")
    print(f"  Ambos son llamados con start_day_idx={day_idx} y rate_eval/loads_eval.")
    print(f"  (rate_eval[{day_idx}] = full_rate_stack[{TRAIN_DAYS}+{day_idx}="
          f"{TRAIN_DAYS+day_idx}])")
    # Detectar si hay algún argumento adicional que pudiera diferirlos
    print("  Ambos están en el mismo bucle for s in range(num_nodes).")
    print("  → H1 DESCARTADA a priori (mismo argumento; confirmar si persiste bug).")

    # ── 3. Reward matrix del día y verificación de matrices ──────────────────
    sep("3. Primeros 5 valores de la diagonal + arcos de ambas rutas")
    rm_unpen, rm_pen = build_day_matrices(
        rate_eval[day_idx], loads_eval[day_idx], distance_arr, diesel_arr
    )
    diag_unpen = [float(rm_unpen.iloc[i, i]) for i in range(5)]
    diag_pen   = [float(rm_pen.iloc[i, i])   for i in range(5)]
    print(f"  reward_matrix (sin penalizar) diagonal[0:4] = {diag_unpen}")
    print(f"  reward_matrix (penalizada)    diagonal[0:4] = {diag_pen}")

    print("\n  Arcos ruta DRL Det  [15,11,13,1,15] — rm_unpen vs rm_pen:")
    for (i, j) in [(15, 11), (11, 13), (13, 1), (1, 15)]:
        v_u = float(rm_unpen.iloc[i, j])
        v_p = float(rm_pen.iloc[i, j])
        tag = "✓" if v_u == v_p else "✗ DIFIEREN"
        print(f"    ({i}→{j}): unpen={v_u:8.1f}  pen={v_p:8.1f}  {tag}")

    print("\n  Arcos ruta MIP-Exact [15,12,16,1,15] — rm_unpen vs rm_pen:")
    for (i, j) in [(15, 12), (12, 16), (16, 1), (1, 15)]:
        v_u = float(rm_unpen.iloc[i, j])
        v_p = float(rm_pen.iloc[i, j])
        tag = "✓" if v_u == v_p else "✗ DIFIEREN"
        print(f"    ({i}→{j}): unpen={v_u:8.1f}  pen={v_p:8.1f}  {tag}")

    # H2: off-diagonal identidad global
    sep("H2 — ¿Las matrices unpen vs pen difieren fuera de la diagonal?")
    diffs_offdiag = []
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i == j:
                continue
            if rm_unpen.iloc[i, j] != rm_pen.iloc[i, j]:
                diffs_offdiag.append((i, j, float(rm_unpen.iloc[i, j]),
                                      float(rm_pen.iloc[i, j])))
    if diffs_offdiag:
        print(f"  ✗ {len(diffs_offdiag)} diferencias off-diagonal encontradas:")
        for (i, j, u, p) in diffs_offdiag[:5]:
            print(f"    ({i},{j}): unpen={u}  pen={p}")
        h2_confirmed = True
    else:
        print("  ✓ Todos los valores off-diagonal son idénticos entre ambas matrices.")
        h2_confirmed = False
    print(f"  → H2 {'CONFIRMADA' if h2_confirmed else 'DESCARTADA'}.")

    # ── 4. Cálculo manual paso a paso ────────────────────────────────────────
    def manual_step_reward(route, label):
        """Replica exactamente simulate_route_reward con avail_prob_arr=None."""
        print(f"\n  Ruta {label}: {route}")
        t_el  = 0.0
        total = 0.0
        for step in range(len(route) - 1):
            i, j = route[step], route[step + 1]
            offset = int(t_el // 14)
            d_idx  = min(day_idx + offset, max_day)
            _, rm_s = build_day_matrices(
                rate_eval[d_idx], loads_eval[d_idx], distance_arr, diesel_arr
            )
            arc_r = float(rm_s.iloc[i, j])
            arc_t = float(time_matrix_np[i, j])
            print(f"    step {step}: ({i}→{j}) t_elapsed={t_el:.2f}h "
                  f"offset={offset} day_used={d_idx} "
                  f"arc_reward={arc_r:.1f} arc_time={arc_t:.2f}h")
            total += arc_r
            t_el  += arc_t
        print(f"    → TOTAL reward={total:.1f}  duration={t_el:.2f}h  "
              f"feasible={t_el <= MAX_DURATION}")
        return total, t_el

    sep("4a. Cálculo manual paso a paso — DRL Det [15,11,13,1,15]")
    r_drl_manual, d_drl_manual = manual_step_reward(DRL_DET_ROUTE, "DRL Det")

    sep("4b. Cálculo manual paso a paso — MIP-Exact [15,12,16,1,15]")
    r_mip_manual, d_mip_manual = manual_step_reward(MIP_EXACT_ROUTE, "MIP-Exact")

    # ── 5. simulate_route_reward oficial ────────────────────────────────────
    sep("5. simulate_route_reward oficial (avail_prob_arr=None)")
    r_drl_det, d_drl_det = simulate_route_reward(
        DRL_DET_ROUTE, START_NODE, day_idx,
        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    r_mip_rte, d_mip_rte = simulate_route_reward(
        MIP_EXACT_ROUTE, START_NODE, day_idx,
        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    print(f"  DRL Det  {DRL_DET_ROUTE}:  "
          f"reward={r_drl_det:.1f}  dur={d_drl_det:.2f}h  "
          f"feasible={d_drl_det <= MAX_DURATION}")
    print(f"  MIP-Rute {MIP_EXACT_ROUTE}: "
          f"reward={r_mip_rte:.1f}  dur={d_mip_rte:.2f}h  "
          f"feasible={d_mip_rte <= MAX_DURATION}")
    print(f"  MAX_DURATION = {MAX_DURATION:.1f}h")

    # ── H3: Bernoulli activo ─────────────────────────────────────────────────
    sep("H3 — ¿El DRL Det fue evaluado con Bernoulli activo?")
    r_drl_stoch, _ = simulate_route_reward(
        DRL_DET_ROUTE, START_NODE, day_idx,
        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
        avail_prob_arr=avail_prob_arr,
    )
    print(f"  simulate_route_reward DRL Det con avail_prob_arr activo: {r_drl_stoch:.1f}")
    print(f"  simulate_route_reward DRL Det sin avail_prob_arr:        {r_drl_det:.1f}")
    print(f"  Reward reportado en el log:                              {DRL_DET_REPORTED_REWARD:.1f}")
    if abs(r_drl_stoch - DRL_DET_REPORTED_REWARD) < 1.0:
        h3_confirmed = True
        print("  ✓ El reward con Bernoulli coincide con el log.")
    elif abs(r_drl_det - DRL_DET_REPORTED_REWARD) < 1.0:
        h3_confirmed = False
        print("  ✓ El reward sin Bernoulli coincide con el log (H3 descartada).")
    else:
        h3_confirmed = None
        print("  ✗ Ninguno coincide: la ruta DRL Det reportada puede ser distinta.")
    print(f"  → H3 {'CONFIRMADA' if h3_confirmed else 'DESCARTADA' if h3_confirmed is False else 'INDETERMINADA'}.")

    # ── H4: Floating-point raw vs rounded ───────────────────────────────────
    sep("H4 — Floating point / np.round en los arcos clave")
    from config import MPG, MARGINAL_COST_SIN_DIESEL
    print("  Comparando reward raw (sin round) vs rounded (np.round(...,0)):")
    for label, route in [("DRL Det ", DRL_DET_ROUTE), ("MIP-Rute", MIP_EXACT_ROUTE)]:
        t_el = 0.0
        for step in range(len(route) - 1):
            i, j = route[step], route[step + 1]
            d_idx = min(day_idx + int(t_el // 14), max_day)
            rev_raw  = rate_eval[d_idx, i, j] * distance_arr[i, j]
            if loads_eval[d_idx, i, j] <= 1:
                rev_raw = 0.0
            cost_raw = (distance_arr[i, j] * (diesel_arr[i, j] / MPG)
                        + distance_arr[i, j] * MARGINAL_COST_SIN_DIESEL)
            raw = rev_raw - cost_raw
            rnd = float(np.round(raw, 0))
            t_el += float(time_matrix_np[i, j])
            tag = "" if abs(raw - rnd) < 0.5 else "  ← rounding gap > 0.5"
            print(f"    {label} ({i}→{j}): raw={raw:10.4f}  rounded={rnd:8.1f}{tag}")

    # ── 6. solve_mip_exact real ───────────────────────────────────────────────
    sep("6. solve_mip_exact (llamada real, mismos args que evaluation.py)")
    print(f"  Ejecutando solve_mip_exact("
          f"start_node={START_NODE}, day_idx={day_idx}, "
          f"max_d={MAX_DURATION}, num_n={NUM_NODES}) ...")
    mip_status, mip_route, mip_reward, mip_dur = solve_mip_exact(
        start_node     = START_NODE,
        time_matrix_np = time_matrix_np,
        rate_stack     = rate_eval,
        loads_stack    = loads_eval,
        distance_arr   = distance_arr,
        diesel_arr     = diesel_arr,
        max_d          = MAX_DURATION,
        num_n          = NUM_NODES,
        start_day_idx  = day_idx,
        avail_prob_arr = None,
    )
    print(f"  MIP status   = {mip_status}")
    print(f"  MIP route    = {mip_route}")
    print(f"  MIP reward   = {mip_reward:.1f}  (via simulate_route_reward interno)")
    print(f"  MIP duration = {mip_dur:.2f}h")
    if mip_route is not None:
        r_mip_reeval, d_mip_reeval = simulate_route_reward(
            mip_route, START_NODE, day_idx,
            time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
            avail_prob_arr=None,
        )
        print(f"  Re-evaluación externa:     {r_mip_reeval:.1f}  dur={d_mip_reeval:.2f}h")
        if abs(mip_reward - r_mip_reeval) > 0.5:
            print("  ✗ solve_mip_exact reward DIFIERE de la re-evaluación externa!")
        else:
            print("  ✓ solve_mip_exact reward == re-evaluación externa.")

    # Calcular el objetivo MIP (R_det unpenalizado) para ambas rutas de forma manual
    sep("6b. Objetivo MIP (R_det, no penalizado) calculado manualmente para ambas rutas")
    K_max = int(MAX_DURATION // 14)
    R_det = {}
    for k in range(K_max + 1):
        d_k = min(day_idx + k, max_day)
        rm_k, _ = build_day_matrices(
            rate_eval[d_k], loads_eval[d_k], distance_arr, diesel_arr
        )
        R_det[k] = rm_k.to_numpy()
        print(f"  R_det[k={k}] -> rate_eval[{d_k}] (offset={k})")

    def mip_obj_value(route):
        """Replica la función objetivo del MIP usando R_det y tiempo acumulado."""
        t_el = 0.0
        obj  = 0.0
        for step in range(len(route) - 1):
            i, j = route[step], route[step + 1]
            k = min(int(t_el // 14), K_max)
            v = R_det[k][i, j]
            obj += v
            t_el += float(time_matrix_np[i, j])
        return obj

    obj_drl = mip_obj_value(DRL_DET_ROUTE)
    obj_mip = mip_obj_value(MIP_EXACT_ROUTE)

    print(f"\n  Objetivo MIP para DRL Det  {DRL_DET_ROUTE}: {obj_drl:.1f}")
    print(f"  Objetivo MIP para MIP-Rute {MIP_EXACT_ROUTE}: {obj_mip:.1f}")
    if mip_route and mip_route not in (DRL_DET_ROUTE, MIP_EXACT_ROUTE):
        obj_mip_real = mip_obj_value(mip_route)
        print(f"  Objetivo MIP para ruta real {mip_route}: {obj_mip_real:.1f}")

    print(f"\n  simulate_route_reward para DRL Det:   {r_drl_det:.1f}")
    print(f"  simulate_route_reward para MIP-Rute:  {r_mip_rte:.1f}")

    delta_drl = obj_drl - r_drl_det
    delta_mip = obj_mip - r_mip_rte
    print(f"\n  Δ(obj_MIP - sim_reward) para DRL Det:   {delta_drl:+.1f}")
    print(f"  Δ(obj_MIP - sim_reward) para MIP-Rute:  {delta_mip:+.1f}")

    if abs(delta_drl) > 1.0 or abs(delta_mip) > 1.0:
        print("\n  ✗ El objetivo del MIP y simulate_route_reward DAN VALORES DISTINTOS.")
        print("    Esto significa que el MIP toma decisiones sobre un paisaje de reward")
        print("    diferente al que usa simulate_route_reward para reportar resultados.")
    else:
        print("\n  ✓ El objetivo del MIP y simulate_route_reward son consistentes.")

    # ── 6c. Verificar objetivo CBC real vs simulate_route_reward ─────────────
    sep("6c. CBC objetivo real (pulp.value) vs simulate_route_reward del MIP")
    print("  Ejecutando MIP inline para capturar pulp.value(prob.objective) ...")
    import pulp
    _nodes       = list(range(NUM_NODES))
    _other_nodes = [n for n in _nodes if n != START_NODE]
    _BIG_M_T = float(MAX_DURATION) + float(time_matrix_np.max())
    _BIG_M_D = float(MAX_DURATION)
    _EPS     = 1e-6

    _prob = pulp.LpProblem(f"_Diag_{START_NODE}", pulp.LpMaximize)
    _x = {(i, j): pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
          for i in _nodes for j in _nodes if i != j}
    _t = {i: pulp.LpVariable(f"t_{i}", lowBound=0.0, upBound=float(MAX_DURATION))
          for i in _nodes}
    _z = {(i, j, k): pulp.LpVariable(f"z_{i}_{j}_{k}", cat=pulp.LpBinary)
          for i in _nodes for j in _nodes if i != j
          for k in range(K_max + 1)}

    _prob += pulp.lpSum(
        float(R_det[k][i, j]) * _z[(i, j, k)]
        for i in _nodes for j in _nodes if i != j
        for k in range(K_max + 1)
    )
    _prob += pulp.lpSum(_x[(START_NODE, j)] for j in _other_nodes) == 1
    _prob += pulp.lpSum(_x[(j, START_NODE)] for j in _other_nodes) == 1
    for i in _other_nodes:
        _prob += (pulp.lpSum(_x[(i, j)] for j in _nodes if j != i) ==
                  pulp.lpSum(_x[(j, i)] for j in _nodes if j != i))
        _prob += pulp.lpSum(_x[(j, i)] for j in _nodes if j != i) <= 1
    _prob += (pulp.lpSum(float(time_matrix_np[i][j]) * _x[(i, j)]
                         for i in _nodes for j in _nodes if i != j) <= MAX_DURATION)
    _prob += _t[START_NODE] == 0.0
    for j in _other_nodes:
        for i in _nodes:
            if i == j:
                continue
            _tij = float(time_matrix_np[i][j])
            _prob += _t[j] >= _t[i] + _tij - _BIG_M_T * (1 - _x[(i, j)])
            _prob += _t[j] <= _t[i] + _tij + _BIG_M_T * (1 - _x[(i, j)])
    for i in _nodes:
        for j in _nodes:
            if i == j:
                continue
            _prob += pulp.lpSum(_z[(i, j, k)] for k in range(K_max + 1)) == _x[(i, j)]
    for i in _nodes:
        for j in _nodes:
            if i == j:
                continue
            for k in range(K_max + 1):
                _prob += _t[i] >= 14.0*k        - _BIG_M_D * (1 - _z[(i, j, k)])
                _prob += _t[i] <= 14.0*(k+1) - _EPS + _BIG_M_D * (1 - _z[(i, j, k)])

    pulp.PULP_CBC_CMD(msg=0, timeLimit=600).solve(_prob)
    _status = pulp.LpStatus[_prob.status]
    _cbc_obj = pulp.value(_prob.objective)
    print(f"  CBC status    = {_status}")
    print(f"  CBC objective = {_cbc_obj:.1f}  (lo que el solver declara como óptimo)")

    # Reconstruir ruta del MIP inline
    _cur = START_NODE
    _route_inline = [START_NODE]
    _moved = True
    while _moved:
        _moved = False
        for j in _nodes:
            if j == _cur:
                continue
            v = _x[(_cur, j)].varValue
            if v is not None and v > 0.99:
                _route_inline.append(j)
                _cur = j
                _moved = True
                break
        if _cur == START_NODE and len(_route_inline) > 1:
            break
    print(f"  Ruta inline   = {_route_inline}")
    _r_inline, _d_inline = simulate_route_reward(
        _route_inline, START_NODE, day_idx,
        time_matrix_np, rate_eval, loads_eval, distance_arr, diesel_arr,
        avail_prob_arr=None,
    )
    print(f"  sim_reward(ruta inline) = {_r_inline:.1f}  dur={_d_inline:.2f}h")
    _obj_inflated = (_cbc_obj is not None and abs(_cbc_obj - _r_inline) > 1.0)
    if _obj_inflated:
        print(f"\n  ✗ INFLADO: CBC obj={_cbc_obj:.1f} ≠ sim_reward={_r_inline:.1f}  "
              f"(delta={_cbc_obj - _r_inline:.1f})")
        print("    → El MIP maximiza un objetivo INCONSISTENTE con simulate_route_reward.")
        print("    → Causa: z variables pueden asignar day-slots más rentables que el")
        print("      tiempo secuencial real permite.")
    else:
        print(f"\n  ✓ CBC obj ≈ sim_reward: no hay inflación del objetivo.")
        _drl_obj_lt_cbc = (_cbc_obj is not None and _cbc_obj < r_drl_det - 1.0)
        if _drl_obj_lt_cbc:
            print(f"  ✗ PERO: CBC obj={_cbc_obj:.1f} < DRL Det sim={r_drl_det:.1f}.")
            print(f"    → El MIP devuelve una solución SUBÓPTIMA. Posibles causas:")
            print(f"    · Timeout (600s) alcanzado antes de encontrar la ruta óptima.")
            print(f"    · Bug de formulación que hace [15,11,13,1,15] infeasible en el MIP.")
            # Check DRL route feasibility in MIP explicitly via duration constraint
            _dur_drl = sum(float(time_matrix_np[DRL_DET_ROUTE[s]][DRL_DET_ROUTE[s+1]])
                           for s in range(len(DRL_DET_ROUTE)-1))
            print(f"\n    Duración ruta DRL Det: {_dur_drl:.2f}h  MAX_DURATION={MAX_DURATION:.1f}h")
            print(f"    ¿Factible por duración? {_dur_drl <= MAX_DURATION}")
        else:
            print(f"  ✓ CBC obj ≈ DRL Det. MIP encontró la ruta correcta (o similar).")

    # ── DIAGNÓSTICO FINAL ────────────────────────────────────────────────────
    sep("DIAGNÓSTICO FINAL")

    bug_reproducido = (
        mip_route is not None
        and r_drl_det > mip_reward + 1.0
    )
    print(f"  ¿Bug reproducido? {bug_reproducido}")
    if bug_reproducido:
        print(f"    simulate_route_reward(DRL Det)  = {r_drl_det:.1f}")
        print(f"    solve_mip_exact reward           = {mip_reward:.1f}")
        print(f"    Δ (DRL - MIP)                   = {r_drl_det - mip_reward:.1f}")

    print()
    h1_verdict = "DESCARTADA"
    h2_verdict = "CONFIRMADA" if h2_confirmed else "DESCARTADA"
    h3_verdict = ("CONFIRMADA" if h3_confirmed
                  else "DESCARTADA" if h3_confirmed is False
                  else "INDETERMINADA")
    h4_verdict = "VER ARRIBA"

    print(f"  H1 (day_idx distinto):            {h1_verdict}")
    print(f"  H2 (matrices distintas off-diag): {h2_verdict}")
    print(f"  H3 (Bernoulli activo en DRL Det): {h3_verdict}")
    print(f"  H4 (floating point / np.round):   {h4_verdict}")

    if obj_drl - r_drl_det > 1.0 or obj_mip - r_mip_rte > 1.0 or (
        mip_route and abs(mip_obj_value(mip_route) - mip_reward) > 1.0
    ):
        print()
        print("  CAUSA RAÍZ PROBABLE:")
        print("  El MIP optimiza con R_det[k] (1ª devolución de build_day_matrices)")
        print("  pero simulate_route_reward usa la 2ª devolución.")
        print("  Si los valores off-diagonal difieren (improbable pero posible),")
        print("  el MIP elige rutas que son subóptimas bajo simulate_route_reward.")
        print()
        print("  Alternativamente: la asignación de day-slot en el MIP (vía variables")
        print("  continuas t[] y restricciones BIG_M) puede divergir del cómputo")
        print("  secuencial de simulate_route_reward si hay arcos con tiempos cercanos")
        print("  a múltiplos de 14 y el solver introduce drift numérico.")

    elif bug_reproducido and obj_mip > obj_drl:
        print()
        print("  CAUSA RAÍZ PROBABLE:")
        print(f"  El MIP ve a {MIP_EXACT_ROUTE} como óptimo (obj={obj_mip:.1f})")
        print(f"  pero simulate_route_reward da {r_mip_rte:.1f} < {r_drl_det:.1f}.")
        print("  Hay una discrepancia entre el objetivo del MIP y la evaluación final.")

    elif bug_reproducido and h3_confirmed:
        print()
        print("  CAUSA RAÍZ: H3 — DRL Det evaluado con Bernoulli activo.")
        print(f"  La ruta {DRL_DET_ROUTE} con avail_prob_arr activo da {r_drl_stoch:.1f},")
        print(f"  que coincide con el log ({DRL_DET_REPORTED_REWARD:.1f}).")
        print(f"  Sin Bernoulli (correcto): {r_drl_det:.1f}.")

    elif not bug_reproducido:
        print()
        print("  El bug NO se reproduce con el day_idx y rutas indicados.")
        print("  Posibles causas externas:")
        print("  · La ruta DRL Det real puede diferir de [15,11,13,1,15].")
        print("  · El MIP puede diferir de [15,12,16,1,15] en otro run.")
        print("  · Verificar el run original con las rutas exactas del log.")

    sep("POSIBLE FIX (si bug confirmado)")
    if bug_reproducido:
        print("  Según el diagnóstico, el fix más probable es uno de:")
        print()
        if _obj_inflated:
            print("  CAUSA RAÍZ: H2/H4 — El objetivo del MIP (z*R_det) es inconsistente")
            print("  con simulate_route_reward. Las variables z pueden asignarse a day-slots")
            print("  más rentables que los que la propagación secuencial de tiempo daría.")
            print()
            print("  FIX: Reformular solve_mip_exact para que el objetivo MIP y")
            print("  simulate_route_reward sean bit-exact.")
            print("  Archivo: Solvers.py  función: solve_mip_exact")
            print("  Opción A — Eliminar las variables z y fijar la recompensa de cada arco")
            print("    usando solo la matriz del día de inicio (R_det[0]). El MIP perdería")
            print("    la capacidad de modelar días dinámicos, pero sería consistente.")
            print("  Opción B — Reemplazar R_det[k] por la diferencia de reward entre el")
            print("    día k y el día base, asegurando que la asignación de z refleje la")
            print("    propagación temporal exacta de simulate_route_reward.")
        elif (_cbc_obj is not None and _cbc_obj < r_drl_det - 1.0):
            _dur_drl = sum(float(time_matrix_np[DRL_DET_ROUTE[s]][DRL_DET_ROUTE[s+1]])
                           for s in range(len(DRL_DET_ROUTE)-1))
            if _dur_drl <= MAX_DURATION:
                print("  CAUSA RAÍZ: El MIP devuelve una solución SUBÓPTIMA.")
                print("  La ruta DRL Det es feasible en duración pero el MIP no la encontró.")
                print()
                print("  FIX MÁS PROBABLE — Aumentar el timeLimit del solver CBC:")
                print("  Archivo: Solvers.py  función: solve_mip_exact  (~línea 1678)")
                print("  Línea:   pulp.PULP_CBC_CMD(msg=0, timeLimit=600).solve(prob)")
                print("  Fix:     pulp.PULP_CBC_CMD(msg=0, timeLimit=3600).solve(prob)")
                print("  O para debug/diagnóstico: quitar timeLimit completamente.")
                print()
                print("  FIX ALTERNATIVO — Añadir solución inicial (warm start) con la ruta")
                print("  greedy para guiar el B&B hacia el óptimo más rápidamente.")
        elif h3_confirmed:
            print("  Archivo: evaluation.py  función: run_solver_comparison")
            print("  Buscar la llamada a simulate_route_reward para DRL Det (~línea 147).")
            print("  Fix:  cambiar 'avail_prob_arr=avail_prob_arr'")
            print("        a      'avail_prob_arr=None'")
        else:
            print("  Ver sección 6c para la causa raíz identificada.")
    else:
        print("  Bug no reproducido en este run. Ver diagnóstico de hipótesis arriba.")

    sep()
    print("FIN DEL DIAGNÓSTICO")


if __name__ == "__main__":
    main()
