"""
evaluation.py
=============
Post-training evaluation:
  · generate_optimal_route  — deterministic greedy rollout with trained agent
  · evaluate_stochastic     — N-episode stochastic reward distribution
  · run_solver_comparison   — DRL vs Greedy vs 2-Opt vs GA vs LNS
  · save_results            — Excel output
  · plot_diagnostics        — 2×2 training diagnostics figure
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    MAX_DURATION,
    REWARD_SCALE_FACTOR,
    N_EVAL_EPISODES, SEED, TRAIN_DAYS,
    get_beam_width_real,
)
from problem_data import build_day_matrices
from Solvers import (
    solve_HGA_LNS_metaheuristic,
    solve_label_setting_exact,
    solve_label_setting_oracle,
    solve_heuristic_rolling_horizon,
    solve_heuristic_rolling_horizon_lookahead,
    solve_heuristic_rolling_horizon_lookahead_stochastic,
    solve_heuristic_rolling_horizon_stochastic,
    solve_mc_rollout_stochastic,
    simulate_route_reward,
)


# ── DRL env-based rollout ─────────────────────────────────────
def rollout_drl_env(
    agent,
    start_node:     int,
    start_day_idx:  int,
    time_matrix,
    rate_stack:     np.ndarray,
    loads_stack:    np.ndarray,
    distance_arr:   np.ndarray,
    diesel_arr:     np.ndarray,
    max_duration:   float      = MAX_DURATION,
    ltr_stack:      np.ndarray = None,
    trucks_stack:   np.ndarray = None,
    avail_prob_arr: np.ndarray = None,
    beam_width:     int        = None,
    critic                     = None,
    value_coef:     float      = 1.0,
) -> tuple:
    """Beam search con días de mercado dinámicos.

    Delega en agent.beam_search_dynamic(): las decisiones y la recompensa acumulada
    usan la matriz del día corriente según el time_elapsed de cada beam.
    """
    agent.eval()
    try:
        return agent.beam_search_dynamic(
            start_node, start_day_idx,
            time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
            max_duration,
            ltr_stack=ltr_stack, trucks_stack=trucks_stack,
            avail_prob_arr=avail_prob_arr,
            beam_width=beam_width,
            critic=critic, value_coef=value_coef,
        )
    finally:
        agent.train()


# ── Solver comparison ─────────────────────────────────────────
def run_solver_comparison(agent, time_matrix,
                           rate_stack, loads_stack, distance_arr, diesel_arr,
                           num_nodes,
                           ltr_stack=None, trucks_stack=None, avail_prob_arr=None,
                           n_days_per_node=3, critic=None):
    """Run DRL Real + DRL Det + HGA-LNS + RH-Greedy for every start node.

    Para cada nodo de inicio se usan n_days_per_node días del set de evaluación
    (días TRAIN_DAYS … total-1), sin repetición dentro del mismo nodo. Los índices
    se pre-generan con un RNG dedicado antes del loop para que no dependan del
    estado random del agente.
    """
    results          = []
    drl_real_times   = []
    drl_det_times    = []
    hga_lns_times    = []
    ls_exact_times   = []
    ls_oracle_times  = []
    rh_greedy_times  = []
    rh_lookahead_times = []
    rh_stoch_times   = []
    rh_lookahead_stoch_times = []
    mc_rollout_times = []
    num_days         = rate_stack.shape[0]

    rng = np.random.default_rng(SEED)
    if n_days_per_node > num_days:
        raise ValueError(
            f"n_days_per_node={n_days_per_node} excede num_days={num_days} disponibles"
        )
    day_indices = np.array([
        rng.choice(num_days, size=n_days_per_node, replace=False)
        for _ in range(num_nodes)
    ])

    time_matrix_np = np.array(time_matrix, dtype=float)

    print("\n--- Solver Comparison (eval set) ---")
    print(f"Eval days: {num_days} | absolute range: [{TRAIN_DAYS}, {TRAIN_DAYS + num_days - 1}]")
    print(f"Day assignments per node: {day_indices.tolist()}\n")

    for s in range(num_nodes):
        for rep in range(n_days_per_node):
            day_idx     = int(day_indices[s, rep])
            abs_day_idx = TRAIN_DAYS + day_idx
            print(f"\nStart node {s}/{num_nodes-1} | rep {rep}/{n_days_per_node-1} | eval day {day_idx} (abs day {abs_day_idx})", flush=True)
            row = {'Start Node': s, 'Rep': rep, 'Eval Day Index': day_idx, 'Abs Day Index': abs_day_idx}

            # ── DRL Real — rollout único bajo revelación post-decisión (mundo fijo por seed) ──
            t0 = time.time()
            drl_real_route, drl_real_reward, drl_real_duration = rollout_drl_env(
                agent, s, day_idx,
                time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                avail_prob_arr=avail_prob_arr,
                beam_width=get_beam_width_real(num_nodes),
                critic=critic,
            )
            drl_real_times.append(time.time() - t0)
            drl_real_valid = drl_real_route is not None
            row.update({
                'DRL Real Route':    drl_real_route,
                'DRL Real Reward':   drl_real_reward   if drl_real_valid else -np.inf,
                'DRL Real Duration': drl_real_duration if drl_real_valid else np.inf,
                'DRL Real Valid':    drl_real_valid,
            })
            _p_drl_real = f"  DRL Real:       {drl_real_route} | reward {drl_real_reward:.1f}"

            # ── DRL Det — ruta construida y evaluada sin Bernoulli ───────────────
            # avail_prob_arr=None desactiva el filtrado de lanes en beam_search_dynamic.
            # El reward se recomputa explícitamente con simulate_route_reward para
            # garantizar paridad bit-a-bit con el mundo determinista.
            t0 = time.time()
            drl_det_route, _, drl_det_duration_raw = rollout_drl_env(
                agent, s, day_idx,
                time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                avail_prob_arr=None,   # sin Bernoulli → mundo determinista
                critic=critic,
            )
            drl_det_valid = drl_det_route is not None
            if drl_det_valid:
                drl_det_reward, drl_det_duration = simulate_route_reward(
                    drl_det_route, s, day_idx,
                    time_matrix_np, rate_stack, loads_stack, distance_arr, diesel_arr,
                    avail_prob_arr=None,
                )
            else:
                drl_det_reward, drl_det_duration = -np.inf, np.inf
            drl_det_times.append(time.time() - t0)
            row.update({
                'DRL Det Route':    drl_det_route,
                'DRL Det Reward':   drl_det_reward   if drl_det_valid else -np.inf,
                'DRL Det Duration': drl_det_duration if drl_det_valid else np.inf,
                'DRL Det Valid':    drl_det_valid,
            })
            _p_drl_det = f"  DRL Det:        {drl_det_route} | reward {drl_det_reward:.1f}"

            # ── RH-Greedy — greedy miope con día dinámico ────────────────────────
            t0 = time.time()
            rh_status, rh_route, rh_reward, rh_duration, rh_valid = \
                solve_heuristic_rolling_horizon(
                    s, time_matrix, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                    start_day_idx=day_idx,
                )
            rh_greedy_times.append(time.time() - t0)
            row.update({
                'RH-Greedy Route':    rh_route,
                'RH-Greedy Reward':   rh_reward if rh_valid else -np.inf,
                'RH-Greedy Duration': rh_duration if rh_route else np.inf,
                'RH-Greedy Valid':    rh_valid,
            })
            _p_rh_greedy = f"  RH-Greedy:      {rh_route} | reward {rh_reward:.1f}"

            # ── RH-Lookahead — rolling horizon con lookahead determinista ─────────
            t0 = time.time()
            rhl_status, rhl_route, rhl_reward, rhl_duration, rhl_valid = \
                solve_heuristic_rolling_horizon_lookahead(
                    s, time_matrix, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                    start_day_idx=day_idx, lookahead=3,
                )
            rh_lookahead_times.append(time.time() - t0)
            row.update({
                'RH-Lookahead Route':    rhl_route,
                'RH-Lookahead Reward':   rhl_reward if rhl_valid else -np.inf,
                'RH-Lookahead Duration': rhl_duration if rhl_route else np.inf,
                'RH-Lookahead Valid':    rhl_valid,
            })
            _p_rh_lookahead = f"  RH-Lookahead:   {rhl_route} | reward {rhl_reward:.1f}"

            # ── RH-Greedy Real — greedy miope en mundo estocástico ───────────────
            t0 = time.time()
            rh_stoch_status, rh_stoch_route, rh_stoch_reward, rh_stoch_duration, rh_stoch_valid = \
                solve_heuristic_rolling_horizon_stochastic(
                    s, time_matrix, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                    start_day_idx=day_idx, avail_prob_arr=avail_prob_arr,
                )
            rh_stoch_times.append(time.time() - t0)
            row.update({
                'RH-Greedy Real Route':    rh_stoch_route,
                'RH-Greedy Real Reward':   rh_stoch_reward if rh_stoch_valid else -np.inf,
                'RH-Greedy Real Duration': rh_stoch_duration if rh_stoch_route else np.inf,
                'RH-Greedy Real Valid':    rh_stoch_valid,
            })
            _p_rh_greedy_real = f"  RH-Greedy Real: {rh_stoch_route} | reward {rh_stoch_reward:.1f}"

            # ── RH-Lookahead Real — lookahead en mundo estocástico ────────────────
            t0 = time.time()
            rhlr_status, rhlr_route, rhlr_reward, rhlr_duration, rhlr_valid = \
                solve_heuristic_rolling_horizon_lookahead_stochastic(
                    s, time_matrix, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                    start_day_idx=day_idx, avail_prob_arr=avail_prob_arr, lookahead=3,
                )
            rh_lookahead_stoch_times.append(time.time() - t0)
            row.update({
                'RH-Lookahead Real Route':    rhlr_route,
                'RH-Lookahead Real Reward':   rhlr_reward if rhlr_valid else -np.inf,
                'RH-Lookahead Real Duration': rhlr_duration if rhlr_route else np.inf,
                'RH-Lookahead Real Valid':    rhlr_valid,
            })
            _p_rh_lookahead_real = f"  RH-Lookahead Real: {rhlr_route} | reward {rhlr_reward:.1f}"

            # ── MC-Rollout estocástico ────────────────────────────────────────────
            t0 = time.time()
            mc_status, mc_route, mc_reward, mc_duration, mc_valid = \
                solve_mc_rollout_stochastic(
                    s, time_matrix, rate_stack, loads_stack,
                    distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                    start_day_idx=day_idx, avail_prob_arr=avail_prob_arr,
                )
            mc_rollout_times.append(time.time() - t0)
            row.update({
                'MC-Rollout Route':    mc_route,
                'MC-Rollout Reward':   mc_reward if mc_valid else -np.inf,
                'MC-Rollout Duration': mc_duration if mc_route else np.inf,
                'MC-Rollout Valid':    mc_valid,
            })
            _p_mc_rollout = f"  MC-Rollout:     {mc_route} | reward {mc_reward:.1f}"

            # ── HGA-LNS ──────────────────────────────────────────────────────────
            print(f"  [HGA-LNS]...", end=" ", flush=True)
            t0 = time.time()
            hga_status, hga_route, hga_reward, hga_duration = solve_HGA_LNS_metaheuristic(
                s, time_matrix, MAX_DURATION, num_nodes,
                rate_stack=rate_stack, loads_stack=loads_stack,
                distance_arr=distance_arr, diesel_arr=diesel_arr,
                start_day_idx=day_idx,
                seed=SEED + s * 1000 + rep,
            )
            hga_lns_times.append(time.time() - t0)
            print(f"{hga_lns_times[-1]:.1f}s", flush=True)
            row.update({
                'HGA-LNS Status':   hga_status,
                'HGA-LNS Route':    hga_route,
                'HGA-LNS Reward':   hga_reward   if hga_status == 'Optimal' else -np.inf,
                'HGA-LNS Duration': hga_duration if hga_status == 'Optimal' else np.inf,
                'HGA-LNS Valid':    hga_status == 'Optimal' and hga_route is not None,
            })
            _p_hga_lns = f"  HGA-LNS:        {hga_route} | reward {hga_reward:.1f}"

            # ── Label-Setting Exact ───────────────────────────────────────────────
            print(f"  [LS-Exact]...", end=" ", flush=True)
            t0 = time.time()
            ls_status, ls_route, ls_reward, ls_duration = solve_label_setting_exact(
                s, time_matrix_np, rate_stack, loads_stack,
                distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                start_day_idx=day_idx,
                time_limit_seconds=300,
            )
            ls_exact_times.append(time.time() - t0)
            print(f"{ls_exact_times[-1]:.1f}s", flush=True)
            ls_valid = ls_status in ("Optimal", "Time-Limited") and ls_route is not None
            row.update({
                'LS-Exact Status':   ls_status,
                'LS-Exact Route':    ls_route,
                'LS-Exact Reward':   ls_reward   if ls_valid else -np.inf,
                'LS-Exact Duration': ls_duration if ls_valid else np.inf,
                'LS-Exact Valid':    ls_valid,
            })
            _p_ls_exact = f"  LS-Exact:       {ls_route} | reward {ls_reward:.1f} [{ls_status}]"

            # ── Label Setting Oracle (cota clarividente, mundo estocástico) ───────────
            print(f"  [LS-Oracle]...", end=" ", flush=True)
            t0 = time.time()
            lso_status, lso_route, lso_reward, lso_duration = solve_label_setting_oracle(
                s, time_matrix_np, rate_stack, loads_stack,
                distance_arr, diesel_arr, MAX_DURATION, num_nodes,
                start_day_idx=day_idx,
                avail_prob_arr=avail_prob_arr,
                time_limit_seconds=300,
            )
            ls_oracle_times.append(time.time() - t0)
            print(f"{ls_oracle_times[-1]:.1f}s", flush=True)
            lso_valid = lso_status in ("Optimal", "Time-Limited") and lso_route is not None
            row.update({
                'LS-Oracle Status':   lso_status,
                'LS-Oracle Route':    lso_route,
                'LS-Oracle Reward':   lso_reward   if lso_valid else -np.inf,
                'LS-Oracle Duration': lso_duration if lso_valid else np.inf,
                'LS-Oracle Valid':    lso_valid,
            })
            assert lso_reward >= row.get('DRL Real Reward', -np.inf) - 1e-3, (
                f"LS-Oracle ({lso_reward:.1f}) no debería ser superado por DRL Real "
                f"({row.get('DRL Real Reward', float('nan')):.1f}) en start={s} day={day_idx} "
                f"— revisar alineación de semillas Bernoulli."
            )
            _p_ls_oracle = f"  LS-Oracle:      {lso_route} | reward {lso_reward:.1f} [{lso_status}]"

            print("  -- Determinísticos --")
            print(_p_drl_det)
            print(_p_rh_greedy)
            print(_p_rh_lookahead)
            print(_p_hga_lns)
            print(_p_ls_exact)
            print("  -- Estocásticos --")
            print(_p_drl_real)
            print(_p_rh_greedy_real)
            print(_p_rh_lookahead_real)
            print(_p_mc_rollout)
            print(_p_ls_oracle)

            results.append(row)

    df = pd.DataFrame(results)

    timing = {
        'drl_real_times':    drl_real_times,
        'drl_det_times':     drl_det_times,
        'hga_lns_times':     hga_lns_times,
        'ls_exact_times':    ls_exact_times,
        'rh_greedy_times':            rh_greedy_times,
        'rh_lookahead_times':         rh_lookahead_times,
        'rh_stoch_times':             rh_stoch_times,
        'rh_lookahead_stoch_times':   rh_lookahead_stoch_times,
        'mc_rollout_times':           mc_rollout_times,
        'ls_oracle_times':            ls_oracle_times,
    }
    return df, timing


# ── Output ────────────────────────────────────────────────────
def save_results(results_df, summary_rows, output_path):
    from excel_report import build_formatted_excel
    node_size = summary_rows[0]["Node Size"] if summary_rows else None
    build_formatted_excel(results_df, summary_rows, output_path, node_size)
    print(f"Results saved → {output_path}")


def plot_ppo_diagnostics(training_log: list, num_nodes: int, output_path: str):
    """Gráfica 2×3 con métricas internas de PPO por update.

    Métricas graficadas
    -------------------
    Fila 1 : Reward promedio | Explained Variance | Entropía
    Fila 2 : Policy Loss + Value Loss | KL Divergence | Clip Fraction
    """
    try:
        import pandas as pd
        df = pd.DataFrame(training_log)

        fig, axes = plt.subplots(2, 4, figsize=(24, 10))
        fig.suptitle(
            f'PPO Internal Diagnostics — {num_nodes} nodes', fontsize=14
        )

        updates = df["update"].values

        def _plot(ax, y, title, ylabel, color, hline=None, hline_label=None):
            ax.plot(updates, y, linewidth=0.9, color=color)
            ax.set_title(title); ax.set_xlabel("PPO Update"); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if hline is not None:
                ax.axhline(hline, color="red", linestyle="--", linewidth=0.8,
                           label=hline_label or f"{hline}")
                ax.legend(fontsize=8)

        # greedy_reward / greedy_valid_rate solo se registran cada log_freq
        # updates — el resto queda NaN en el DataFrame. dropna() evita que la
        # línea se corte en cada hueco.
        greedy_df = df.dropna(subset=["greedy_reward"]) if "greedy_reward" in df else None

        ax_reward = axes[0, 0]
        ax_reward.plot(updates, df["reward"], linewidth=0.9, color="steelblue", label="Sampled reward")
        if greedy_df is not None and not greedy_df.empty:
            ax_reward.plot(greedy_df["update"], greedy_df["greedy_reward"],
                           linewidth=1.4, color="darkorange", marker="o", markersize=3,
                           label="Greedy reward")
        ax_reward.set_title("Avg Reward per Update"); ax_reward.set_xlabel("PPO Update")
        ax_reward.set_ylabel("Reward"); ax_reward.grid(True, alpha=0.3); ax_reward.legend(fontsize=8)

        _plot(axes[0, 1], df["explained_var"], "Critic Explained Variance", "Expl. Var.", "mediumseagreen",
              hline=0.5, hline_label="threshold 0.5")
        _plot(axes[0, 2], df["entropy"],       "Policy Entropy",          "Entropy",      "mediumpurple")

        ax_valid = axes[0, 3]
        if greedy_df is not None and not greedy_df.empty:
            ax_valid.plot(greedy_df["update"], greedy_df["greedy_valid_rate"],
                          linewidth=1.4, color="teal", marker="o", markersize=3)
            ax_valid.set_ylim(-0.05, 1.05)
        ax_valid.set_title("Greedy Valid Route Rate"); ax_valid.set_xlabel("PPO Update")
        ax_valid.set_ylabel("Valid Rate"); ax_valid.grid(True, alpha=0.3)

        ax_loss = axes[1, 0]
        ax_loss.plot(updates, df["policy_loss"], linewidth=0.9, color="coral",    label="Policy loss")
        ax_loss.plot(updates, df["value_loss"],  linewidth=0.9, color="goldenrod", label="Value loss")
        ax_loss.set_title("Policy & Value Loss"); ax_loss.set_xlabel("PPO Update")
        ax_loss.set_ylabel("Loss"); ax_loss.grid(True, alpha=0.3); ax_loss.legend(fontsize=8)

        _plot(axes[1, 1], df["kl_divergence"], "Approx KL Divergence",   "KL",           "tomato",
              hline=0.02, hline_label="target 0.02")
        _plot(axes[1, 2], df["clip_fraction"], "PPO Clip Fraction",       "Clip Frac.",   "darkorange",
              hline=0.1, hline_label="ref 0.10")
        axes[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"PPO diagnostics plot saved → {output_path}")
    except Exception as e:
        print(f"Warning: could not generate PPO diagnostics plot: {e}")


def plot_diagnostics(episode_rewards, episode_losses, results_df,
                     num_nodes, output_path):
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'DRL Training Diagnostics — {num_nodes} nodes', fontsize=14)

        window = max(1, min(100, len(episode_rewards) // 5))

        ax1 = axes[0, 0]
        ax1.plot(pd.Series(episode_rewards).rolling(window).mean(),
                 linewidth=0.8, color='steelblue')
        ax1.set_title('Training rewards (rolling avg)')
        ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(pd.Series(episode_losses).rolling(window).mean(),
                 linewidth=0.8, color='coral')
        ax2.set_title('Training loss (rolling avg)')
        ax2.set_xlabel('Episode'); ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        nodes = list(range(num_nodes))
        drl_det_r = [
            results_df.loc[(results_df['Start Node'] == n) & results_df['DRL Det Valid'], 'DRL Det Reward'].mean()
            if 'DRL Det Valid' in results_df.columns
            and results_df.loc[results_df['Start Node'] == n, 'DRL Det Valid'].any()
            else 0
            for n in nodes
        ]
        hga_r = [
            results_df.loc[(results_df['Start Node'] == n) & results_df['HGA-LNS Valid'], 'HGA-LNS Reward'].mean()
            if results_df.loc[results_df['Start Node'] == n, 'HGA-LNS Valid'].any()
            else 0
            for n in nodes
        ]
        x     = np.arange(num_nodes); w = 0.35
        ax3.bar(x - w/2, hga_r,     w, label='HGA-LNS', color='forestgreen', alpha=0.8)
        ax3.bar(x + w/2, drl_det_r, w, label='DRL Det',  color='steelblue',   alpha=0.8)
        ax3.set_title('DRL Det vs HGA-LNS per start node')
        ax3.set_xlabel('Start node'); ax3.set_ylabel('Reward')
        ax3.set_xticks(x); ax3.legend(); ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        drl_real_r = [
            results_df.loc[(results_df['Start Node'] == n) & results_df['DRL Real Valid'], 'DRL Real Reward'].mean()
            if 'DRL Real Valid' in results_df.columns
            and results_df.loc[results_df['Start Node'] == n, 'DRL Real Valid'].any()
            else 0
            for n in nodes
        ]
        ax4.bar(x - w/2, drl_real_r, w, label='DRL Real', color='coral',     alpha=0.8)
        ax4.bar(x + w/2, drl_det_r,  w, label='DRL Det',  color='steelblue', alpha=0.8)
        ax4.set_title('DRL Real vs DRL Det per start node')
        ax4.set_xlabel('Start node'); ax4.set_ylabel('Reward')
        ax4.set_xticks(x); ax4.legend(); ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Diagnostics plot saved → {output_path}")
    except Exception as e:
        print(f"Warning: could not generate diagnostics plot: {e}")
