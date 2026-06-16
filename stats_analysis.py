"""
stats_analysis.py
==================
Análisis de robustez estadística sobre el panel de evaluación estocástica
(mundo real con Bernoulli). Compara DRL Real contra cada baseline estocástico
(RH-Greedy Real, RH-Lookahead Real, MC-Rollout) usando:
  · Descriptivos: media, std, percentiles P25/P50/P75, min, max — sobre
    instancias donde el método es válido.
  · Test de Wilcoxon pareado (signed-rank) sobre instancias donde AMBOS
    métodos de la comparación son válidos simultáneamente.

Fundamento metodológico: Demšar (2006, JMLR) y García et al. (2010) recomiendan
tests no paramétricos pareados para comparar heurísticas/políticas sobre el
mismo conjunto de instancias, evitando la asunción de normalidad que requiere
un t-test pareado — relevante aquí porque N es modesto y los rewards pueden
tener outliers (instancias infactibles, nodos con reward muy negativo).
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

STOCHASTIC_METHODS = {
    'DRL Real':          ('DRL Real Reward',          'DRL Real Valid'),
    'RH-Greedy Real':    ('RH-Greedy Real Reward',    'RH-Greedy Real Valid'),
    'RH-Lookahead Real': ('RH-Lookahead Real Reward', 'RH-Lookahead Real Valid'),
    'MC-Rollout':        ('MC-Rollout Reward',        'MC-Rollout Valid'),
}

BASELINE_REFERENCE = 'DRL Real'


def _check_columns(results_df: pd.DataFrame, methods: dict) -> None:
    """Raise KeyError with a clear message if any required column is missing."""
    missing = []
    for name, (rcol, vcol) in methods.items():
        for col in (rcol, vcol):
            if col not in results_df.columns:
                missing.append(f"'{col}' (needed for method '{name}')")
    if missing:
        raise KeyError(
            f"stats_analysis: columnas faltantes en results_df:\n  " +
            "\n  ".join(missing)
        )


def compute_descriptive_stats(results_df: pd.DataFrame,
                               methods: dict = STOCHASTIC_METHODS) -> pd.DataFrame:
    """Media, std, percentiles por método, sobre instancias válidas."""
    _check_columns(results_df, methods)
    rows = []
    for name, (rcol, vcol) in methods.items():
        vals = results_df.loc[results_df[vcol], rcol]
        rows.append({
            'Method':       name,
            'N Valid':      len(vals),
            'N Total':      len(results_df),
            'Mean':         vals.mean(),
            'Std':          vals.std(ddof=1) if len(vals) > 1 else np.nan,
            'P25':          vals.quantile(0.25) if len(vals) > 0 else np.nan,
            'Median (P50)': vals.quantile(0.50) if len(vals) > 0 else np.nan,
            'P75':          vals.quantile(0.75) if len(vals) > 0 else np.nan,
            'Min':          vals.min() if len(vals) > 0 else np.nan,
            'Max':          vals.max() if len(vals) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def compute_wilcoxon_paired(results_df: pd.DataFrame,
                             methods: dict = STOCHASTIC_METHODS,
                             reference: str = BASELINE_REFERENCE,
                             alpha: float = 0.05) -> pd.DataFrame:
    """Wilcoxon pareado: reference vs cada otro método, sobre instancias
    donde ambos son válidos simultáneamente."""
    _check_columns(results_df, methods)
    ref_rcol, ref_vcol = methods[reference]
    rows = []
    for name, (rcol, vcol) in methods.items():
        if name == reference:
            continue
        mask = results_df[ref_vcol] & results_df[vcol]
        n_pairs = int(mask.sum())
        ref_vals   = results_df.loc[mask, ref_rcol].to_numpy()
        other_vals = results_df.loc[mask, rcol].to_numpy()
        diff = ref_vals - other_vals

        if n_pairs < 1 or np.allclose(diff, 0):
            stat, p = np.nan, np.nan
        else:
            try:
                stat, p = wilcoxon(ref_vals, other_vals)
            except ValueError:
                stat, p = np.nan, np.nan

        rows.append({
            'Comparison':              f'{reference} vs {name}',
            'N Pairs':                 n_pairs,
            'Mean Diff (Ref - Other)': diff.mean() if n_pairs > 0 else np.nan,
            'Median Diff':             np.median(diff) if n_pairs > 0 else np.nan,
            'Std Diff':                diff.std(ddof=1) if n_pairs > 1 else np.nan,
            'Wilcoxon Statistic':      stat,
            'p-value':                 p,
            f'Significant (p<{alpha})': bool(p < alpha) if not np.isnan(p) else None,
        })
    return pd.DataFrame(rows)


def append_stats_sheets_to_excel(excel_path: str, results_df: pd.DataFrame,
                                  methods: dict = STOCHASTIC_METHODS,
                                  reference: str = BASELINE_REFERENCE) -> None:
    """Agrega 'Stochastic Descriptive Stats' y 'Wilcoxon Significance' al
    Excel existente sin tocar las hojas previas."""
    desc_df = compute_descriptive_stats(results_df, methods)
    wil_df  = compute_wilcoxon_paired(results_df, methods, reference)

    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        desc_df.to_excel(writer, sheet_name='Stochastic Descriptive Stats', index=False)
        wil_df.to_excel(writer, sheet_name='Wilcoxon Significance', index=False)

    print(f"Hojas estadísticas agregadas → {excel_path}")
