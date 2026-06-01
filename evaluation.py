"""
evaluation.py
=============
Post-training evaluation:
  · generate_optimal_route  — deterministic greedy rollout with trained agent
  · evaluate_stochastic     — N-episode stochastic reward distribution
  · run_solver_comparison   — DRL vs MIP vs Greedy vs 2-Opt vs GA vs LNS
  · save_results            — Excel output
  · plot_diagnostics        — 2×2 training diagnostics figure
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    STOCHASTIC_MODE, MAX_DURATION,
    REWARD_SCALE_FACTOR,
    N_EVAL_EPISODES, NOISE_FRACTION, SEED, TRAIN_DAYS,
)
from problem_data import sample_stochastic_reward, build_day_matrices
from Solvers import (
    solve_mip, solve_heuristic, solve_2opt_heuristic,
    solve_LNS_metaheuristic, solve_genetic_algorithm,
    solve_HGA_LNS_metaheuristic,
)


# ── Route generation ─────────────────────────────────────────
def generate_optimal_route(agent, start_node, time_matrix, reward_matrix_penalized,
                            num_nodes, max_duration=MAX_DURATION,
                            distance_arr=None):
    """Greedy rollout with the trained AMRoutingAgent (no grad).

    Delegates to agent.generate_route() without explicit beam_width —
    el agente resuelve el beam_width dinámicamente vía get_beam_width(num_nodes):
    modelos pequeños (≤10 nodos) usan beam=5, tamaños intermedios (≤35) usan beam=3,
    y modelos grandes usan beam=1 (política robusta, greedy suficiente).
    """
    route, reward, duration = agent.generate_route(
        start_node, reward_matrix_penalized, time_matrix,
        distance_arr, max_duration,
    )
    return route, reward, duration


# ── Stochastic evaluation ─────────────────────────────────────
def evaluate_stochastic(agent, start_node, time_matrix, reward_matrix_penalized,
                         num_nodes, noise_sigma, n_episodes=N_EVAL_EPISODES,
                         distance_arr=None):
    rewards   = []
    durations = []
    valid     = 0

    for _ in range(n_episodes):
        route, reward, duration = generate_optimal_route(
            agent, start_node, time_matrix, reward_matrix_penalized, num_nodes,
            distance_arr=distance_arr)
        if route is not None:
            if STOCHASTIC_MODE and noise_sigma > 0:
                reward = reward + np.random.normal(0, noise_sigma)
            rewards.append(reward)
            durations.append(duration)
            if duration <= MAX_DURATION:
                valid += 1

    if not rewards:
        return {'mean_reward': -np.inf, 'std_reward': 0,
                'median_reward': -np.inf, 'valid_fraction': 0,
                'mean_duration': np.inf, 'n_routes': 0}

    return {
        'mean_reward':    np.mean(rewards),
        'std_reward':     np.std(rewards),
        'median_reward':  np.median(rewards),
        'p10_reward':     np.percentile(rewards, 10),
        'p90_reward':     np.percentile(rewards, 90),
        'valid_fraction': valid / len(rewards),
        'mean_duration':  np.mean(durations),
        'n_routes':       len(rewards),
    }


# ── Solver comparison ─────────────────────────────────────────
def run_solver_comparison(agent, time_matrix,
                           rate_stack, loads_stack, distance_arr, diesel_arr,
                           noise_sigma, num_nodes):
    """Run DRL + all benchmark solvers for every start node.

    Para cada nodo de inicio se usa un día del set de evaluación (días
    TRAIN_DAYS … total-1). Los índices se pre-generan con un RNG dedicado
    antes del loop para que no dependan del estado random del agente, lo que
    garantiza que DRL y todos los solvers compiten sobre exactamente la misma
    realización de mercado y que los resultados son idénticos en cada corrida.
    """
    results           = []
    mip_times         = []
    heuristic_times   = []
    drl_times         = []
    heuristic2_times  = []
    ga_times          = []
    lns_times         = []
    hga_lns_times     = []
    num_days          = rate_stack.shape[0]   # tamaño del set de evaluación

    # Pre-generar índices con RNG propio — aislado del estado random del agente
    rng         = np.random.default_rng(SEED)
    day_indices = rng.integers(0, num_days, size=num_nodes)

    print("\n--- Solver Comparison (eval set) ---")
    print(f"Eval days: {num_days} | absolute range: [{TRAIN_DAYS}, {TRAIN_DAYS + num_days - 1}]")
    print(f"Day assignments per node: {day_indices.tolist()}\n")

    for s in range(num_nodes):
        day_idx     = int(day_indices[s])
        abs_day_idx = TRAIN_DAYS + day_idx   # índice absoluto en el stack original
        reward_matrix, reward_matrix_penalized = build_day_matrices(
            rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
        )
        print(f"\nStart node {s} | eval day {day_idx} (abs day {abs_day_idx})")
        row = {'Start Node': s, 'Eval Day Index': day_idx, 'Abs Day Index': abs_day_idx}

        # DRL — Det evaluation (sin ruido, mismas condiciones que los baselines)
        # Se usa noise_sigma=0 implícitamente: generate_optimal_route es determinista.
        # Esto permite un gap de calidad justo. La evaluación estocástica se reporta aparte.
        t0 = time.time()
        drl_route, drl_reward, drl_duration = generate_optimal_route(
            agent, s, time_matrix, reward_matrix_penalized, num_nodes,
            distance_arr=distance_arr)
        drl_times.append(time.time() - t0)
        row.update({
            'DRL Route':          drl_route,
            'DRL Det Reward':  drl_reward   if drl_route else -np.inf,
            'DRL Duration':       drl_duration if drl_route else np.inf,
            'DRL Valid':          drl_route is not None and drl_route[0] == drl_route[-1],
        })
        # DRL — Stochastic evaluation (con ruido, mide robustez bajo incertidumbre)
        stoch = evaluate_stochastic(agent, s, time_matrix, reward_matrix_penalized,
                                     num_nodes, noise_sigma,
                                     distance_arr=distance_arr)
        row.update({
            'DRL Stoch Mean':   stoch['mean_reward'],
            'DRL Stoch Std':    stoch['std_reward'],
            'DRL Stoch Valid%': stoch['valid_fraction'] * 100,
        })
        print(f"  DRL Det: {drl_route} | reward {drl_reward:.1f} | stoch mean {stoch['mean_reward']:.1f} ± {stoch['std_reward']:.1f}")

        # MIP
        t0 = time.time()
        mip_status, mip_route, mip_reward, mip_duration = solve_mip(
            s, time_matrix, reward_matrix_penalized, MAX_DURATION, num_nodes)
        mip_times.append(time.time() - t0)
        row.update({
            'MIP Status':   mip_status,
            'MIP Route':    mip_route,
            'MIP Reward':   mip_reward   if mip_status == 'Optimal' else -np.inf,
            'MIP Duration': mip_duration if mip_status == 'Optimal' else np.inf,
            'MIP Valid':    mip_status == 'Optimal' and mip_route is not None,
        })
        print(f"  MIP:    {mip_route} | reward {mip_reward:.1f}")

        # Greedy
        t0 = time.time()
        heu_status, heu_route, heu_reward, heu_duration, heu_valid = solve_heuristic(
            s, time_matrix, reward_matrix_penalized, MAX_DURATION, num_nodes)
        heuristic_times.append(time.time() - t0)
        row.update({
            'Heuristic Route':    heu_route,
            'Heuristic Reward':   heu_reward if heu_valid else -np.inf,
            'Heuristic Duration': heu_duration if heu_route else np.inf,
            'Heuristic Valid':    heu_valid,
        })
        print(f"  Greedy: {heu_route} | reward {heu_reward:.1f}")

        # 2-Opt
        t0 = time.time()
        opt_status, opt_route, opt_reward, opt_duration, opt_valid = solve_2opt_heuristic(
            s, time_matrix, reward_matrix_penalized, MAX_DURATION, num_nodes)
        heuristic2_times.append(time.time() - t0)
        row.update({
            '2Opt Route':    opt_route,
            '2Opt Reward':   opt_reward if opt_valid else -np.inf,
            '2Opt Duration': opt_duration if opt_route else np.inf,
            '2Opt Valid':    opt_valid,
        })
        print(f"  2-Opt:  {opt_route} | reward {opt_reward:.1f}")

        # Genetic Algorithm
        t0 = time.time()
        ga_status, ga_route, ga_reward, ga_duration = solve_genetic_algorithm(
            s, time_matrix, reward_matrix_penalized, MAX_DURATION, num_nodes)
        ga_times.append(time.time() - t0)
        row.update({
            'GA Status':   ga_status,
            'GA Route':    ga_route,
            'GA Reward':   ga_reward   if ga_status == 'Optimal' else -np.inf,
            'GA Duration': ga_duration if ga_status == 'Optimal' else np.inf,
            'GA Valid':    ga_status == 'Optimal' and ga_route is not None,
        })
        print(f"  GA:     {ga_route} | reward {ga_reward:.1f}")

        # LNS
        t0 = time.time()
        lns_status, lns_route, lns_reward, lns_duration = solve_LNS_metaheuristic(
            s, time_matrix, reward_matrix_penalized, MAX_DURATION, num_nodes)
        lns_times.append(time.time() - t0)
        row.update({
            'LNS Status':   lns_status,
            'LNS Route':    lns_route,
            'LNS Reward':   lns_reward   if lns_status == 'Optimal' else -np.inf,
            'LNS Duration': lns_duration if lns_status == 'Optimal' else np.inf,
            'LNS Valid':    lns_status == 'Optimal' and lns_route is not None,
        })
        print(f"  LNS:    {lns_route} | reward {lns_reward:.1f}")

        # HGA-LNS
        t0 = time.time()
        hga_status, hga_route, hga_reward, hga_duration = solve_HGA_LNS_metaheuristic(
            s, time_matrix, reward_matrix_penalized, MAX_DURATION, num_nodes)
        hga_lns_times.append(time.time() - t0)
        row.update({
            'HGA-LNS Status':   hga_status,
            'HGA-LNS Route':    hga_route,
            'HGA-LNS Reward':   hga_reward   if hga_status == 'Optimal' else -np.inf,
            'HGA-LNS Duration': hga_duration if hga_status == 'Optimal' else np.inf,
            'HGA-LNS Valid':    hga_status == 'Optimal' and hga_route is not None,
        })
        print(f"  HGA-LNS:{hga_route} | reward {hga_reward:.1f}")

        # Optimality gaps vs MIP — usa DRL Det Reward para comparación justa
        mip_r = row['MIP Reward']
        def gap(solver_r, solver_valid):
            if row['MIP Valid'] and solver_valid and abs(mip_r) > 1e-6:
                return ((mip_r - solver_r) / abs(mip_r)) * 100
            return float('nan')

        row['DRL Gap (%)']       = gap(row['DRL Det Reward'],    row['DRL Valid'])
        row['Heuristic Gap (%)'] = gap(row['Heuristic Reward'], row['Heuristic Valid'])
        row['2Opt Gap (%)']      = gap(row['2Opt Reward'],      row['2Opt Valid'])
        row['GA Gap (%)']        = gap(row['GA Reward'],        row['GA Valid'])
        row['LNS Gap (%)']       = gap(row['LNS Reward'],       row['LNS Valid'])
        row['HGA-LNS Gap (%)']   = gap(row['HGA-LNS Reward'],  row['HGA-LNS Valid'])
        row['DRL Stoch Gap (%)'] = gap(row['DRL Stoch Mean'],  row['DRL Stoch Valid%'] > 0)
        results.append(row)

    df = pd.DataFrame(results)

    timing = {
        'mip_times': mip_times, 'heuristic_times': heuristic_times,
        'drl_times': drl_times, 'heuristic2_times': heuristic2_times,
        'ga_times': ga_times,   'lns_times': lns_times,
        'hga_lns_times': hga_lns_times,
    }
    return df, timing


# ── Output ────────────────────────────────────────────────────
def save_results(results_df, summary_rows, output_path):
    summary_df = pd.DataFrame(summary_rows)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Per Node Results', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
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

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f'PPO Internal Diagnostics — {num_nodes} nodes '
            f'(sigma={NOISE_FRACTION*100:.0f}%)', fontsize=14
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

        _plot(axes[0, 0], df["reward"],       "Avg Reward per Update",   "Reward",       "steelblue")
        _plot(axes[0, 1], df["explained_var"], "Critic Explained Variance", "Expl. Var.", "mediumseagreen",
              hline=0.5, hline_label="threshold 0.5")
        _plot(axes[0, 2], df["entropy"],       "Policy Entropy",          "Entropy",      "mediumpurple")

        ax_loss = axes[1, 0]
        ax_loss.plot(updates, df["policy_loss"], linewidth=0.9, color="coral",    label="Policy loss")
        ax_loss.plot(updates, df["value_loss"],  linewidth=0.9, color="goldenrod", label="Value loss")
        ax_loss.set_title("Policy & Value Loss"); ax_loss.set_xlabel("PPO Update")
        ax_loss.set_ylabel("Loss"); ax_loss.grid(True, alpha=0.3); ax_loss.legend(fontsize=8)

        _plot(axes[1, 1], df["kl_divergence"], "Approx KL Divergence",   "KL",           "tomato",
              hline=0.02, hline_label="target 0.02")
        _plot(axes[1, 2], df["clip_fraction"], "PPO Clip Fraction",       "Clip Frac.",   "darkorange",
              hline=0.1, hline_label="ref 0.10")

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
            f'Stochastic Optimised DRL — {num_nodes} nodes '
            f'(sigma={NOISE_FRACTION*100:.0f}%)', fontsize=14)

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
        drl_r = [results_df.loc[results_df['Start Node'] == n, 'DRL Det Reward'].values[0]
                 if results_df.loc[results_df['Start Node'] == n, 'DRL Valid'].values[0] else 0
                 for n in nodes]
        mip_r = [results_df.loc[results_df['Start Node'] == n, 'MIP Reward'].values[0]
                 if results_df.loc[results_df['Start Node'] == n, 'MIP Valid'].values[0] else 0
                 for n in nodes]
        x     = np.arange(num_nodes); w = 0.35
        ax3.bar(x - w/2, mip_r, w, label='MIP', color='forestgreen', alpha=0.8)
        ax3.bar(x + w/2, drl_r, w, label='DRL', color='steelblue',   alpha=0.8)
        ax3.set_title('DRL vs MIP per start node')
        ax3.set_xlabel('Start node'); ax3.set_ylabel('Reward')
        ax3.set_xticks(x); ax3.legend(); ax3.grid(True, alpha=0.3)

        ax4      = axes[1, 1]
        gap_data = results_df.loc[
            results_df['MIP Valid'] & results_df['DRL Valid'], 'DRL Gap (%)'
        ].dropna()
        if len(gap_data) > 0:
            ax4.bar(range(len(gap_data)), gap_data.values, color='salmon', alpha=0.8)
            ax4.axhline(gap_data.mean(), color='red', linestyle='--',
                        label=f'Avg: {gap_data.mean():.1f}%')
            ax4.legend()
        ax4.set_title('DRL optimality gap vs MIP (%)')
        ax4.set_xlabel('Start node'); ax4.set_ylabel('Gap (%)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Diagnostics plot saved → {output_path}")
    except Exception as e:
        print(f"Warning: could not generate diagnostics plot: {e}")
