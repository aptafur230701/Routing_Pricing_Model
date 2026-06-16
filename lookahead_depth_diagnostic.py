"""
lookahead_depth_diagnostic.py
==============================
Diagnóstico de profundidad de anticipación: corre RH-Lookahead Real con
varios valores de `lookahead` sobre el mismo conjunto de instancias
(start_node, day_idx) ya usado en el panel de 20 nodos, y compara cada
variante contra DRL Real con Wilcoxon pareado.

Objetivo: determinar si el empate estadístico DRL Real vs RH-Lookahead Real
(lookahead=3, p≈0.147) se mantiene, se cierra, o se revierte a medida que el
horizonte de anticipación del baseline aumenta. Esto separa dos hipótesis:
  (a) DRL Real generaliza a anticipación profunda → el empate persiste o se
      mantiene incluso con lookahead alto.
  (b) El empate actual es contra un horizonte todavía corto → la ventaja de
      RH-Lookahead Real crece y supera a DRL Real con más profundidad.

ADVERTENCIA DE COSTO: la implementación de lookahead es búsqueda exhaustiva
sin memoización, O(N^(lookahead+1)) por paso. Sobre N=20 esto puede volverse
prohibitivo rápidamente. Este script mide tiempo por instancia ANTES de
correr el panel completo en cada profundidad, y aborta el siguiente nivel
de profundidad si el tiempo proyectado excede un presupuesto razonable.

Uso
---
  python lookahead_depth_diagnostic.py

Salida
------
  results_20nodes/Lookahead_Depth_Diagnostic_N20.xlsx
    · Timing Sweep      — costo medido y proyectado por nivel de lookahead
    · Wilcoxon vs Depth — p-value y estadístico de DRL Real vs cada variante
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from config import MAX_DURATION, SEED, TRAIN_DAYS
from Solvers import solve_heuristic_rolling_horizon_lookahead_stochastic

# ── Parámetros del diagnóstico ────────────────────────────────
NUM_NODES        = 20
N_DAYS_PER_NODE  = 3   # debe coincidir con N_DAYS_PER_NODE de main.py (panel actual)
LOOKAHEAD_VALUES = [3, 4, 5, 6]
TIME_BUDGET_SEC  = 600  # presupuesto máximo por nivel de profundidad (s)
N_PROBE_INSTANCES = 3   # instancias de sondeo antes de proyectar coste total


# ── Helpers internos ──────────────────────────────────────────

def _reconstruct_instances(num_nodes: int, num_days: int,
                            n_days_per_node: int, seed: int) -> list[tuple[int, int]]:
    """Reproduce exactamente los (start_node, day_idx) que genera run_solver_comparison.

    run_solver_comparison construye day_indices con:
        rng = np.random.default_rng(SEED)
        day_indices = np.array([rng.choice(num_days, size=n_days_per_node, replace=False)
                                 for _ in range(num_nodes)])
    y luego itera for s in range(num_nodes): for rep in range(n_days_per_node).
    """
    rng = np.random.default_rng(seed)
    day_indices = np.array([
        rng.choice(num_days, size=n_days_per_node, replace=False)
        for _ in range(num_nodes)
    ])
    return [
        (s, int(day_indices[s, rep]))
        for s in range(num_nodes)
        for rep in range(n_days_per_node)
    ]


def _run_drl_real_baseline(agent, instances: list, time_matrix, rate_stack,
                            loads_stack, distance_arr, diesel_arr,
                            num_nodes: int, avail_prob_arr,
                            ltr_stack=None, trucks_stack=None) -> pd.DataFrame:
    """Corre DRL Real sobre todas las instancias y construye el results_df base.

    draw_lane_availability usa semilla determinista (start_day_idx, node, arrival_day),
    por lo que los resultados son bit-exactos respecto al panel original.
    """
    from evaluation import rollout_drl_env
    from config import get_beam_width_real

    beam_w = get_beam_width_real(num_nodes)
    rows = []
    n = len(instances)
    print(f"\n[DRL Real baseline] {n} instancias, beam_width={beam_w}")
    t_total = time.time()
    for i, (s, day_idx) in enumerate(instances):
        route, reward, duration = rollout_drl_env(
            agent, s, day_idx,
            time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=avail_prob_arr,
            beam_width=beam_w,
        )
        valid = route is not None
        rows.append({
            'Start Node':     s,
            'Eval Day Index': day_idx,
            'DRL Real Reward': reward if valid else -np.inf,
            'DRL Real Valid':  valid,
        })
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"  [{i+1}/{n}] node={s} day={day_idx} "
                  f"reward={reward:.1f} valid={valid}", flush=True)

    elapsed = time.time() - t_total
    print(f"[DRL Real baseline] completado en {elapsed:.1f}s "
          f"({elapsed/n:.2f}s/instancia)\n")
    return pd.DataFrame(rows)


# ── Funciones del sweep ───────────────────────────────────────

def probe_cost(lookahead: int, instances: list,
               time_matrix, rate_stack, loads_stack,
               distance_arr, diesel_arr, num_nodes: int,
               avail_prob_arr) -> tuple[float, float]:
    """Sondea N_PROBE_INSTANCES instancias y proyecta el coste total."""
    probe = instances[:N_PROBE_INSTANCES]
    t0 = time.time()
    for s, day_idx in probe:
        solve_heuristic_rolling_horizon_lookahead_stochastic(
            s, time_matrix, rate_stack, loads_stack,
            distance_arr, diesel_arr, MAX_DURATION, num_nodes,
            start_day_idx=day_idx, avail_prob_arr=avail_prob_arr,
            lookahead=lookahead,
        )
    elapsed = time.time() - t0
    per_instance = elapsed / len(probe)
    projected_total = per_instance * len(instances)
    return per_instance, projected_total


def run_lookahead_sweep(results_df: pd.DataFrame,
                        time_matrix, rate_stack, loads_stack,
                        distance_arr, diesel_arr,
                        num_nodes: int, avail_prob_arr,
                        lookahead_values: list = None,
                        time_budget_sec: float = TIME_BUDGET_SEC
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Corre RH-Lookahead Real con cada valor de lookahead sobre las mismas
    instancias (Start Node, Eval Day Index) ya presentes en results_df.
    Reutiliza DRL Real Reward / DRL Real Valid ya calculados en results_df —
    no se vuelve a correr DRL.
    """
    if lookahead_values is None:
        lookahead_values = LOOKAHEAD_VALUES

    instances = list(zip(results_df['Start Node'].astype(int),
                         results_df['Eval Day Index'].astype(int)))

    sweep_rows    = []
    wilcoxon_rows = []
    aborted_at    = None

    for lk in lookahead_values:
        print(f"\n{'─'*55}")
        print(f"[lookahead={lk}] Sondeando {N_PROBE_INSTANCES} instancias…")
        per_inst, projected = probe_cost(
            lk, instances, time_matrix, rate_stack, loads_stack,
            distance_arr, diesel_arr, num_nodes, avail_prob_arr,
        )
        print(f"[lookahead={lk}] costo estimado : {per_inst:.2f}s/instancia")
        print(f"[lookahead={lk}] proyectado total: {projected:.1f}s "
              f"sobre {len(instances)} instancias (presupuesto={time_budget_sec}s)")

        if projected > time_budget_sec:
            aborted_at = lk
            print(f"[lookahead={lk}] EXCEDE presupuesto → ABORTANDO este nivel "
                  f"y los siguientes más costosos.")
            print(f"  → Esto es una LIMITACIÓN COMPUTACIONAL, no un resultado: "
                  f"el sweep completó lookahead ∈ "
                  f"{[v for v in lookahead_values if v < lk]} "
                  f"y no pudo evaluar lookahead ≥ {lk} dentro de "
                  f"{time_budget_sec}s por nivel.")
            sweep_rows.append({
                'Lookahead':              lk,
                'N Instances':            len(instances),
                'N Probe':                N_PROBE_INSTANCES,
                'Time/Instance Probe (s)': round(per_inst, 3),
                'Projected Total (s)':    round(projected, 1),
                'Total Time (s)':         None,
                'Mean Time/Instance (s)': None,
                'Status':                 f'ABORTADO (presupuesto={time_budget_sec}s)',
            })
            break

        print(f"[lookahead={lk}] Corriendo panel completo ({len(instances)} instancias)…")
        rewards, valids = [], []
        t0 = time.time()
        for i, (s, day_idx) in enumerate(instances):
            _, _, reward, _, valid = solve_heuristic_rolling_horizon_lookahead_stochastic(
                s, time_matrix, rate_stack, loads_stack,
                distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                start_day_idx=day_idx, avail_prob_arr=avail_prob_arr,
                lookahead=lk,
            )
            rewards.append(reward if valid else -np.inf)
            valids.append(valid)
            if (i + 1) % 10 == 0 or (i + 1) == len(instances):
                print(f"  [{i+1}/{len(instances)}] node={s} day={day_idx} "
                      f"reward={reward:.1f} valid={valid}", flush=True)
        elapsed = time.time() - t0

        col_reward = f'RH-Lookahead{lk} Real Reward'
        col_valid  = f'RH-Lookahead{lk} Real Valid'
        results_df[col_reward] = rewards
        results_df[col_valid]  = valids

        sweep_rows.append({
            'Lookahead':              lk,
            'N Instances':            len(instances),
            'N Probe':                N_PROBE_INSTANCES,
            'Time/Instance Probe (s)': round(per_inst, 3),
            'Projected Total (s)':    round(projected, 1),
            'Total Time (s)':         round(elapsed, 1),
            'Mean Time/Instance (s)': round(elapsed / len(instances), 3),
            'Status':                 'OK',
        })

        # ── Wilcoxon pareado: DRL Real vs este nivel ──────────────
        mask    = results_df['DRL Real Valid'] & results_df[col_valid]
        n_pairs = int(mask.sum())
        drl_vals   = results_df.loc[mask, 'DRL Real Reward'].to_numpy()
        other_vals = results_df.loc[mask, col_reward].to_numpy()
        diff = drl_vals - other_vals

        if n_pairs < 1 or np.allclose(diff, 0):
            stat, p = np.nan, np.nan
        else:
            try:
                stat, p = wilcoxon(drl_vals, other_vals)
            except ValueError:
                stat, p = np.nan, np.nan

        n_valid_drl   = int(results_df['DRL Real Valid'].sum())
        n_valid_other = int(results_df[col_valid].sum())
        mean_drl   = results_df.loc[results_df['DRL Real Valid'], 'DRL Real Reward'].mean()
        mean_other = results_df.loc[results_df[col_valid], col_reward].mean()

        print(f"\n[lookahead={lk}] Wilcoxon DRL Real vs RH-Lookahead{lk} Real")
        print(f"  N pares válidos : {n_pairs}")
        print(f"  Mean DRL Real   : {mean_drl:.2f}  (n_valid={n_valid_drl})")
        print(f"  Mean RH-LK{lk}  : {mean_other:.2f}  (n_valid={n_valid_other})")
        print(f"  Mean diff (DRL−RH-LK{lk}): {diff.mean():.2f}")
        print(f"  p-value         : {p:.4f}  {'→ SIGNIFICATIVO (p<0.05)' if not np.isnan(p) and p < 0.05 else '→ no significativo'}")

        wilcoxon_rows.append({
            'Comparison':               f'DRL Real vs RH-Lookahead{lk} Real',
            'Lookahead':                lk,
            'N Valid DRL Real':         n_valid_drl,
            'N Valid RH-Lookahead':     n_valid_other,
            'N Pairs (both valid)':     n_pairs,
            'Mean DRL Real Reward':     round(mean_drl,   2) if not np.isnan(mean_drl)   else None,
            'Mean RH-Lookahead Reward': round(mean_other, 2) if not np.isnan(mean_other) else None,
            'Mean Diff (DRL - Other)':  round(diff.mean(), 2) if n_pairs > 0 else None,
            'Median Diff':              round(float(np.median(diff)), 2) if n_pairs > 0 else None,
            'Wilcoxon Statistic':       round(float(stat), 3) if not np.isnan(stat) else None,
            'p-value':                  round(float(p),    4) if not np.isnan(p)    else None,
            'Significant (p<0.05)':     bool(p < 0.05) if not np.isnan(p) else None,
            'Status':                   'OK',
        })

    # Registrar niveles abortados en wilcoxon_rows para que quede explícito en la hoja
    if aborted_at is not None:
        for lk in lookahead_values:
            if lk >= aborted_at and not any(r['Lookahead'] == lk for r in wilcoxon_rows):
                wilcoxon_rows.append({
                    'Comparison':           f'DRL Real vs RH-Lookahead{lk} Real',
                    'Lookahead':             lk,
                    'Status':                f'NO EJECUTADO — presupuesto {time_budget_sec}s excedido en lookahead={aborted_at}',
                })

    sweep_df    = pd.DataFrame(sweep_rows)
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)
    return results_df, sweep_df, wilcoxon_df


def _regression_check(wilcoxon_df: pd.DataFrame,
                       expected_p: float = 0.147,
                       expected_mean_diff: float = 52.58,
                       tol_p: float = 0.05,
                       tol_diff: float = 10.0) -> None:
    """Comprueba que lookahead=3 reproduce los valores del panel original.

    Si los números difieren más del umbral, imprime una advertencia explícita —
    indica que las instancias o el avail_prob_arr no coinciden con el panel.
    """
    row3 = wilcoxon_df[wilcoxon_df['Lookahead'] == 3]
    if row3.empty or row3.iloc[0].get('Status', 'OK') != 'OK':
        print("\n[REGRESIÓN] lookahead=3 no fue ejecutado — no se puede validar.")
        return

    r = row3.iloc[0]
    p_got    = r.get('p-value',               None)
    diff_got = r.get('Mean Diff (DRL - Other)', None)

    ok_p    = p_got    is not None and abs(p_got    - expected_p)        <= tol_p
    ok_diff = diff_got is not None and abs(diff_got - expected_mean_diff) <= tol_diff

    print("\n" + "═" * 55)
    print("CHECK DE REGRESIÓN — lookahead=3 vs panel original")
    print("═" * 55)
    print(f"  p-value    : obtenido={p_got}   esperado≈{expected_p}   "
          f"{'✓ OK' if ok_p    else '✗ DIVERGE (tolerancia ±' + str(tol_p) + ')'}")
    print(f"  mean diff  : obtenido={diff_got}  esperado≈{expected_mean_diff}  "
          f"{'✓ OK' if ok_diff else '✗ DIVERGE (tolerancia ±' + str(tol_diff) + ')'}")
    if ok_p and ok_diff:
        print("  → El sweep usa exactamente las mismas instancias y "
              "avail_prob_arr que el panel original. Los niveles 4/5/6 son fiables.")
    else:
        print("  → ADVERTENCIA: los valores no coinciden con el panel original.")
        print("     Verificar: NUM_NODES, N_DAYS_PER_NODE, SEED, TRAIN_DAYS, "
              "avail_prob_arr y checkpoint.")
    print("═" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────

def main():
    import torch
    from config import DEVICE
    from problem_data import load_matrices
    from am_agent import AMRoutingAgent

    cwd = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cwd, f"results_{NUM_NODES}nodes")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 55)
    print(f"  Lookahead Depth Diagnostic — N={NUM_NODES} nodos")
    print(f"  N_DAYS_PER_NODE={N_DAYS_PER_NODE}  SEED={SEED}")
    print(f"  Lookaheads a evaluar: {LOOKAHEAD_VALUES}")
    print(f"  Presupuesto por nivel: {TIME_BUDGET_SEC}s")
    print("=" * 55)

    # ── Cargar matrices ───────────────────────────────────────
    print("\n[1/4] Cargando matrices…")
    (time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
     ltr_stack, trucks_stack, avail_prob_arr, _) = load_matrices(NUM_NODES)

    rate_eval  = rate_stack[TRAIN_DAYS:]
    loads_eval = loads_stack[TRAIN_DAYS:]
    num_days   = rate_eval.shape[0]
    print(f"      Días de evaluación disponibles: {num_days} "
          f"(días {TRAIN_DAYS}–{TRAIN_DAYS + num_days - 1})")

    # ── Cargar agente ─────────────────────────────────────────
    print("\n[2/4] Cargando agente desde checkpoint…")
    checkpoint_dir = os.path.join(cwd, "checkpoints")
    ckpt_path = os.path.join(checkpoint_dir, f"am_checkpoint_{NUM_NODES}nodes.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint no encontrado: {ckpt_path}\n"
            f"Ejecuta main.py con EVAL_ONLY=False primero, o copia el .pt en checkpoints/."
        )
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    agent = AMRoutingAgent(NUM_NODES, device=DEVICE)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    print(f"      Checkpoint cargado: {ckpt_path}")

    # ── Reconstruir instancias ────────────────────────────────
    print("\n[3/4] Reconstruyendo instancias (start_node, day_idx)…")
    instances = _reconstruct_instances(NUM_NODES, num_days, N_DAYS_PER_NODE, SEED)
    print(f"      Total instancias: {len(instances)}")
    print(f"      Primeras 5: {instances[:5]}")

    # ── DRL Real baseline ─────────────────────────────────────
    results_df = _run_drl_real_baseline(
        agent, instances, time_matrix, rate_eval, loads_eval,
        distance_arr, diesel_arr, NUM_NODES, avail_prob_arr,
        ltr_stack=ltr_stack, trucks_stack=trucks_stack,
    )
    n_drl_valid = int(results_df['DRL Real Valid'].sum())
    print(f"DRL Real baseline: {n_drl_valid}/{len(instances)} instancias válidas, "
          f"mean reward = {results_df.loc[results_df['DRL Real Valid'], 'DRL Real Reward'].mean():.2f}")

    # ── Sweep de lookahead ────────────────────────────────────
    print("\n[4/4] Ejecutando sweep de lookahead…")
    results_df, sweep_df, wilcoxon_df = run_lookahead_sweep(
        results_df,
        time_matrix, rate_eval, loads_eval,
        distance_arr, diesel_arr, NUM_NODES, avail_prob_arr,
        lookahead_values=LOOKAHEAD_VALUES,
        time_budget_sec=TIME_BUDGET_SEC,
    )

    # ── Check de regresión ────────────────────────────────────
    _regression_check(wilcoxon_df)

    # ── Resumen final ─────────────────────────────────────────
    print("\n" + "═" * 55)
    print("RESUMEN DEL SWEEP")
    print("═" * 55)
    print(sweep_df.to_string(index=False))
    print()
    cols_to_show = [c for c in [
        'Comparison', 'Lookahead', 'N Pairs (both valid)',
        'Mean Diff (DRL - Other)', 'p-value', 'Significant (p<0.05)', 'Status',
    ] if c in wilcoxon_df.columns]
    print(wilcoxon_df[cols_to_show].to_string(index=False))

    # ── Guardar Excel ─────────────────────────────────────────
    label    = f"N{NUM_NODES}"
    out_path = os.path.join(out_dir, f"Lookahead_Depth_Diagnostic_{label}.xlsx")
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        sweep_df.to_excel(writer, sheet_name='Timing Sweep', index=False)
        wilcoxon_df.to_excel(writer, sheet_name='Wilcoxon vs Depth', index=False)
        # hoja auxiliar con los rewards por instancia (útil para análisis posterior)
        reward_cols = ['Start Node', 'Eval Day Index', 'DRL Real Reward', 'DRL Real Valid']
        for lk in LOOKAHEAD_VALUES:
            for sfx in (f'RH-Lookahead{lk} Real Reward', f'RH-Lookahead{lk} Real Valid'):
                if sfx in results_df.columns:
                    reward_cols.append(sfx)
        results_df[reward_cols].to_excel(writer, sheet_name='Per Instance Rewards', index=False)

    print(f"\nGuardado → {out_path}")
    print("Hojas: 'Timing Sweep' | 'Wilcoxon vs Depth' | 'Per Instance Rewards'")


if __name__ == "__main__":
    main()
