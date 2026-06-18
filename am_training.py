"""
am_training.py
==============
Loop de entrenamiento PPO para AMRoutingAgent + CriticHead.

Algoritmo: Proximal Policy Optimization (PPO-Clip)
  1. Recolectar N episodios con la política actual (sin gradientes).
  2. Calcular ventajas con GAE (Generalized Advantage Estimation).
  3. Actualizar actor + critic durante K épocas con mini-batches.
  4. Repetir hasta completar el número de updates.

Interfaz análoga a run_training() de training.py:

    agent, critic, episode_rewards, episode_losses = run_am_training(
        time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        num_nodes
    )

· Usa RoutingEnv como entorno estándar (Siguiendo practicas de API Gymnasium).
"""

import numpy as np
import torch
import torch.nn as nn

from config import (
    MAX_DURATION,
    DEVICE,
    SEED,
    TRAIN_DAYS,
    N_EVAL_EPISODES,
    get_episodes_per_node,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF, AM_PRE_NORM, AM_DROPOUT,
    PPO_N_EPISODES_PER_UPDATE, PPO_N_EPOCHS, PPO_BATCH_SIZE,
    PPO_LR, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS,
    PPO_ENTROPY_COEF, PPO_ENTROPY_COEF_START, PPO_ENTROPY_COEF_END, PPO_GRAD_CLIP,
    PPO_WEIGHT_DECAY, PPO_LR_DECAY, PPO_LR_END,
    USE_SELF_CRITICAL, SELF_CRITICAL_COEF, SELF_CRITICAL_WARMUP_UPDATES,
)
from routing_env import RoutingEnv, VectorRoutingEnv
from problem_data import build_day_matrices, build_rm_pen_stack
from attention_encoder import (
    build_node_features_batch,
    build_temporal_features_batch,
    build_market_features_batch,
)
from am_agent import AMRoutingAgent
from critic_head import CriticHead
from debug_utils import check_tensor


# ─────────────────────────────────────────────────────────────────────────────
# Buffer de rollout para PPO
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Almacena transiciones de un ciclo de recolección PPO.

    Cada transición guarda las entradas crudas (numpy) necesarias para
    reconstruir el forward pass durante la fase de actualización.
    """

    def __init__(self):
        self.node_feats    = []   # list of np.ndarray (N, N_NODE_FEATURES)
        self.temporal_feats = []  # list of np.ndarray (3,)
        self.market_feats  = []   # list of np.ndarray (1,)
        self.current_nodes = []   # list of int
        self.masks         = []   # list of np.ndarray (N,) int8
        self.actions       = []   # list of int
        self.rewards       = []   # list of float
        self.old_log_probs = []   # list of float
        self.old_values    = []   # list of float
        self.dones         = []   # list of bool
        # (start_idx, end_idx, last_value, ep_truncated) por episodio
        self.episode_boundaries = []
        # Calculados por compute_gae_per_episode()
        self.returns       = None
        self.advantages    = None
        # Self-critical (Camino B) — poblados solo si USE_SELF_CRITICAL
        self.sc_advantage  = None   # (n_episodes,) ventaja normalizada por episodio
        self.episode_idx   = None   # (n_transitions,) índice de episodio por transición

    def add(
        self,
        node_feats:    np.ndarray,
        temporal:      np.ndarray,
        current_node:  int,
        mask:          np.ndarray,
        action:        int,
        reward:        float,
        log_prob:      float,
        value:         float,
        done:          bool,
        market_feats:  np.ndarray = None,   # (1,)
    ):
        self.node_feats.append(node_feats)
        self.temporal_feats.append(temporal)
        self.market_feats.append(market_feats if market_feats is not None
                                 else np.zeros(1, dtype=np.float32))
        self.current_nodes.append(current_node)
        self.masks.append(mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.old_log_probs.append(log_prob)
        self.old_values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.actions)

    def compute_gae(
        self,
        last_value: float,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Calcula retornos y ventajas con GAE.

        last_value : V(s_T) del último estado — 0.0 si el episodio terminó,
                     V(s_T) del critic si fue truncado.
        """
        n = len(self.rewards)
        returns    = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        values = np.array(self.old_values + [last_value], dtype=np.float32)
        dones  = np.array(self.dones, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(n)):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae   = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t]    = gae + values[t]

        self.returns    = returns
        self.advantages = advantages

    def compute_gae_per_episode(
        self,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Calcula retornos y ventajas con GAE usando el last_value correcto de
        cada episodio registrado en episode_boundaries.

        Cada episodio se procesa en forma independiente: el bootstrap del
        último paso usa last_value si fue truncado, 0 si terminó naturalmente.
        No se usa la máscara dones dentro del segmento porque el loop de
        recolección rompe en done=True, garantizando que ningún paso
        intermedio tenga done=True.
        """
        n = len(self.rewards)
        returns    = np.zeros(n, dtype=np.float32)
        advantages = np.zeros(n, dtype=np.float32)

        for start_idx, end_idx, last_value, ep_truncated in self.episode_boundaries:
            seg_len   = end_idx - start_idx
            bootstrap = last_value if ep_truncated else 0.0

            seg_values = np.array(
                self.old_values[start_idx:end_idx] + [bootstrap],
                dtype=np.float32,
            )

            gae = 0.0
            for t in reversed(range(seg_len)):
                delta = (
                    self.rewards[start_idx + t]
                    + gamma * seg_values[t + 1]
                    - seg_values[t]
                )
                gae                        = delta + gamma * gae_lambda * gae
                advantages[start_idx + t]  = gae
                returns[start_idx + t]     = gae + seg_values[t]

        self.returns    = returns
        self.advantages = advantages

    def get_batches(self, batch_size: int):
        """
        Genera índices de mini-batches aleatorios sobre el buffer completo.
        Si el buffer es menor que batch_size devuelve un solo batch con todo.
        """
        n = len(self)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            yield indices[start: start + batch_size]


# ─────────────────────────────────────────────────────────────────────────────
# Forward pass compartido (recolección y actualización)
# ─────────────────────────────────────────────────────────────────────────────

def _forward(
    agent:          AMRoutingAgent,
    critic:         CriticHead,
    node_feats_t:   torch.Tensor,    # (B, N, N_NODE_FEATURES)
    temporal_t:     torch.Tensor,    # (B, 3)
    current_nodes:  torch.Tensor,    # (B,) int64
    mask_t:         torch.Tensor,    # (B, N) int8
    actions_t:      torch.Tensor = None,   # (B,) int64 — None en recolección
    market_feats_t: torch.Tensor = None,   # (B, 1)
):
    """
    Ejecuta encoder → context → decoder → critic para un batch.

    Si actions_t es None (recolección): samplea acciones nuevas.
    Si actions_t no es None (actualización PPO): evalúa acciones almacenadas.

    Retorna
    -------
    actions   : Tensor (B,) int64
    log_probs : Tensor (B,)
    entropies : Tensor (B,)
    values    : Tensor (B,)
    """
    B = node_feats_t.shape[0]

    check_tensor("node_feats_t", node_feats_t)
    check_tensor("temporal_t", temporal_t)

    embeddings, graph_emb = agent.encoder(node_feats_t)           # (B,N,d_h), (B,d_h)

    check_tensor("embeddings", embeddings)
    check_tensor("graph_emb", graph_emb)

    current_emb = embeddings[torch.arange(B), current_nodes, :]   # (B, d_h)
    check_tensor("current_emb", current_emb)

    h_t = agent.context_net(graph_emb, current_emb, temporal_t, market_feats_t)   # (B, d_h)
    check_tensor("context h_t", h_t)

    values = critic(h_t)                                           # (B,)
    check_tensor("values", values)

    # Validate mask: every sample must have at least one valid action
    valid_actions = mask_t.sum(dim=-1)
    if (valid_actions == 0).any():
        raise RuntimeError(
            f"Invalid mask: sample(s) with zero valid actions detected "
            f"(valid counts per sample: {valid_actions.tolist()})"
        )

    if actions_t is None:
        actions, log_probs, entropies = agent.decoder.act(h_t, embeddings, mask_t)
    else:
        log_probs, entropies = agent.decoder.evaluate_action(
            h_t, embeddings, mask_t, actions_t
        )
        actions = actions_t

    check_tensor("log_probs", log_probs)

    return actions, log_probs, entropies, values


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación greedy periódica (observacional, no afecta gradientes)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_greedy(
    agent,
    eval_start_nodes: np.ndarray,
    eval_start_days:  np.ndarray,
    time_matrix,
    rate_stack:       np.ndarray,
    loads_stack:      np.ndarray,
    distance_arr:     np.ndarray,
    diesel_arr:       np.ndarray,
    ltr_stack:        np.ndarray,
    trucks_stack:     np.ndarray,
    avail_prob_arr:   np.ndarray,
    reward_global_p95: float,
) -> tuple:
    """
    Recompensa promedio en modo greedy (beam_width=1) sobre un set fijo de
    episodios (mismos nodos de inicio y días entre llamadas), para monitorear
    el desempeño en el modo que de verdad se usa en producción — distinto del
    avg_reward de los rollouts de entrenamiento, que samplean de la política.

    beam_search_dynamic devuelve route=None, reward=-np.inf cuando ningún beam
    logra cerrar el ciclo de regreso al depot dentro de max_duration. Esos
    episodios se excluyen del promedio de reward (para no contaminarlo con
    -inf) y se cuentan aparte en valid_rate.

    Retorna
    -------
    avg_reward : float — promedio solo sobre episodios con ruta válida
                 (0.0 si ninguno fue válido).
    valid_rate : float — fracción de episodios con ruta válida.
    """
    agent.eval()
    rewards = []
    n_valid = 0
    try:
        with torch.no_grad():
            for start_node, start_day in zip(eval_start_nodes, eval_start_days):
                route, reward, _ = agent.beam_search_dynamic(
                    int(start_node), int(start_day),
                    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                    MAX_DURATION, beam_width=1,
                    ltr_stack=ltr_stack, trucks_stack=trucks_stack,
                    avail_prob_arr=avail_prob_arr,
                    reward_global_p95=reward_global_p95,
                )
                if route is not None:
                    n_valid += 1
                    rewards.append(reward)
    finally:
        agent.train()
    n_total    = len(eval_start_nodes)
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    valid_rate = n_valid / n_total
    return avg_reward, valid_rate


# ─────────────────────────────────────────────────────────────────────────────
# Rollout greedy vectorizado para el baseline self-critical (Camino B)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_greedy_rollout(
    agent:          AMRoutingAgent,
    vec_env:        VectorRoutingEnv,
    start_nodes_arr: np.ndarray,
    start_day_idxs:  np.ndarray,
    rm_pen_stack:    np.ndarray,
    time_matrix_arr: np.ndarray,
    distance_arr:    np.ndarray,
    trucks_stack:    np.ndarray,
    avail_prob_arr:  np.ndarray,
    reward_global_p95: float,
    ltr_stack:       np.ndarray,
    num_nodes:       int,
) -> np.ndarray:
    """
    Rollout greedy (argmax, beam=1) vectorizado sobre los MISMOS
    (start_node, start_day_idx) del update actual.

    Reutiliza VectorRoutingEnv (el mismo step() que la recolección muestreada)
    en vez de beam_search_dynamic, para que la fórmula de recompensa
    (REWARD_SCALE_FACTOR, bono de éxito, penalización por tiempo, warning
    temporal) y la realización estocástica del mundo (draw_lane_availability,
    determinista en start_day_idx/node/arrival_day) sean IDÉNTICAS a las del
    rollout muestreado — condición necesaria para que la ventaja self-critical
    (R_sample - R_greedy) compare en las mismas unidades.

    Asume que agent/critic ya están en eval() (llamado entre la fase de
    recolección y la fase de update del loop principal). No samplea ni usa
    RNG global: greedy_action es argmax puro, así que no perturba la
    secuencia de np.random/torch usada por el resto del loop.
    """
    B = len(start_nodes_arr)
    ep_rewards = np.zeros(B, dtype=np.float32)

    masks  = vec_env.reset(start_nodes_arr, start_day_idxs)
    active = np.ones(B, dtype=bool)

    with torch.no_grad():
        for _ in range(num_nodes):   # defensive ceiling, igual que la recolección
            if not active.any():
                break

            cur_nodes = vec_env.current_node
            time_el   = vec_env.time_elapsed
            step_cnt  = vec_env.step_count
            day_idx   = vec_env.current_day_idx

            node_feats = build_node_features_batch(
                cur_nodes, start_nodes_arr, vec_env.visited_mask,
                rm_pen_stack, time_matrix_arr, distance_arr,
                day_idx, num_nodes, MAX_DURATION,
                trucks_stack_raw=trucks_stack,
                avail_prob_arr=avail_prob_arr,
                reward_global_p95=reward_global_p95,
            )
            temporal = build_temporal_features_batch(time_el, step_cnt, MAX_DURATION, num_nodes)
            market   = build_market_features_batch(cur_nodes, day_idx, ltr_stack) if ltr_stack is not None \
                       else np.zeros((B, 1), dtype=np.float32)

            node_feats_t = torch.from_numpy(node_feats).to(DEVICE)
            temporal_t   = torch.from_numpy(temporal).to(DEVICE)
            market_t     = torch.from_numpy(market).to(DEVICE)
            cur_nodes_t  = torch.tensor(cur_nodes, dtype=torch.long, device=DEVICE)
            mask_t       = torch.from_numpy(masks).to(DEVICE)

            embeddings, graph_emb = agent.encoder(node_feats_t)
            current_emb = embeddings[torch.arange(B), cur_nodes_t, :]
            h_t = agent.context_net(graph_emb, current_emb, temporal_t, market_t)
            actions_t = agent.decoder.greedy_action(h_t, embeddings, mask_t)
            actions = actions_t.cpu().numpy()

            rewards, terminated, next_masks = vec_env.step(actions)
            ep_rewards += np.where(active, rewards, 0.0)

            active = active & ~terminated
            masks  = next_masks

    return ep_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_am_training(
    time_matrix,
    rate_stack:     np.ndarray,
    loads_stack:    np.ndarray,
    distance_arr:   np.ndarray,
    diesel_arr:     np.ndarray,
    num_nodes:      int,
    pretrained_agent:  AMRoutingAgent = None,
    pretrained_critic: CriticHead     = None,
    ltr_stack:         np.ndarray = None,   # [num_nodes, 120]
    trucks_stack:      np.ndarray = None,   # [num_nodes, 120, 3]
    avail_prob_arr:    np.ndarray = None,   # [num_nodes, num_nodes]
    reward_global_p95: float      = 1.0,
) -> tuple:
    """
    Entrena AMRoutingAgent + CriticHead con PPO.

    Parámetros
    ----------
    time_matrix    : np.ndarray — tiempos entre nodos.
    rate_stack     : np.ndarray [num_days, N, N]
    loads_stack    : np.ndarray [num_days, N, N]
    distance_arr   : np.ndarray [N, N]
    diesel_arr     : np.ndarray [N, N]
    num_nodes      : int
    ltr_stack      : np.ndarray [num_nodes, 120] — LTR por hub y día
    trucks_stack   : np.ndarray [num_nodes, 120, 3] — camiones por hub, día y delta
    avail_prob_arr : np.ndarray [num_nodes, num_nodes] — prior Bernoulli de disponibilidad

    Retorna
    -------
    agent          : AMRoutingAgent entrenado.
    critic         : CriticHead entrenado.
    episode_rewards: list[float]
    episode_losses : list[float]
    """
    d_h                   = AM_D_H
    n_heads               = AM_N_HEADS
    n_layers              = AM_N_LAYERS
    d_ff                  = AM_D_FF
    n_episodes_per_update = PPO_N_EPISODES_PER_UPDATE
    n_ppo_epochs          = PPO_N_EPOCHS
    ppo_batch_size        = PPO_BATCH_SIZE
    lr                    = PPO_LR
    gamma                 = PPO_GAMMA
    gae_lambda            = PPO_GAE_LAMBDA
    clip_eps              = PPO_CLIP_EPS
    grad_clip             = PPO_GRAD_CLIP

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    episodes_per_node = get_episodes_per_node(num_nodes)
    total_episodes    = episodes_per_node * num_nodes
    n_updates         = max(1, total_episodes // n_episodes_per_update)
    num_days          = rate_stack.shape[0]
    num_train_days    = min(TRAIN_DAYS, num_days)

    # ── Modelos ───────────────────────────────────────────────────────────────
    agent  = pretrained_agent  if pretrained_agent  is not None \
             else AMRoutingAgent(
                 num_nodes, d_h, n_heads, n_layers, d_ff, device=DEVICE,
                 pre_norm=AM_PRE_NORM, dropout=AM_DROPOUT,
             )
    critic = pretrained_critic if pretrained_critic is not None \
             else CriticHead(d_h).to(DEVICE)

    # Optimizers separados: el critic necesita converger más rápido que el actor
    # para dar señales de ventaja de calidad. Con un optimizer compartido y lr=3e-5
    # el critic aprende demasiado lento y el EV se estanca por debajo de 0.7.
    # Ratio 10x entre critic y actor es estándar en PPO para problemas combinatorios.
    # Actor con AdamW: weight_decay es el regularizador PPO-safe primario (no
    # corrompe el ratio de importancia como lo haría dropout en el path de logits).
    # Critic se deja en Adam puro (sin weight decay): su objetivo es MSE de
    # regresión, no política, y no comparte el problema de corrupción del ratio.
    actor_optimizer  = torch.optim.AdamW(agent.parameters(),  lr=lr, weight_decay=PPO_WEIGHT_DECAY)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr * 10)

    # ── Precomputar stack de reward matrices (una sola vez, fuera del loop) ──────
    time_matrix_arr = (
        time_matrix.to_numpy(dtype=float)
        if hasattr(time_matrix, "to_numpy")
        else np.asarray(time_matrix, dtype=float)
    )
    rm_pen_stack = build_rm_pen_stack(
        rate_stack[:num_train_days], loads_stack[:num_train_days],
        distance_arr, diesel_arr,
    )   # (num_train_days, N, N) float32

    # ── Entorno vectorizado (una sola instancia, reutilizada en cada update) ──
    vec_env = VectorRoutingEnv(
        time_matrix=time_matrix,
        rate_stack=rate_stack,
        loads_stack=loads_stack,
        distance_arr=distance_arr,
        diesel_arr=diesel_arr,
        avail_prob_arr=avail_prob_arr,
        rm_pen_stack=rm_pen_stack,
        num_nodes=num_nodes,
        max_duration=MAX_DURATION,
    )

    print(
        f"\n--- Entrenamiento AM-PPO: {total_episodes} episodios "
        f"({n_updates} updates × {n_episodes_per_update} ep/update) ---"
    )

    # ── Set fijo de evaluación greedy (mismos episodios en cada chequeo) ──────
    # RandomState propio, separado del RNG global de numpy: así no perturba la
    # secuencia de muestreo de start_nodes/start_day_idxs del loop de entrenamiento.
    n_eval_episodes = min(N_EVAL_EPISODES, num_nodes, 20)
    eval_rng         = np.random.RandomState(SEED)
    eval_start_nodes = eval_rng.choice(num_nodes, size=n_eval_episodes, replace=False)
    eval_start_days  = eval_rng.randint(0, num_train_days, size=n_eval_episodes)

    episode_rewards = []
    episode_losses  = []
    training_log    = []

    # ── Loop principal ────────────────────────────────────────────────────────
    for update in range(n_updates):
        progress     = update / max(1, n_updates - 1)
        entropy_coef = PPO_ENTROPY_COEF_START + (PPO_ENTROPY_COEF_END - PPO_ENTROPY_COEF_START) * progress

        # LR del actor: decae linealmente de PPO_LR a PPO_LR_END si PPO_LR_DECAY,
        # constante en PPO_LR en caso contrario. El critic mantiene el ratio 10x.
        actor_lr = lr + (PPO_LR_END - lr) * progress if PPO_LR_DECAY else lr
        actor_optimizer.param_groups[0]["lr"]  = actor_lr
        critic_optimizer.param_groups[0]["lr"] = actor_lr * 10

        buffer = RolloutBuffer()
        update_ep_rewards = []

        # ── Fase 1: Recolección vectorizada (B episodios en paralelo) ─────────
        agent.eval()
        critic.eval()

        B = n_episodes_per_update   # 360

        # Balanced start-node sampling: each node appears exactly floor(B/N) times,
        # with remainder filled by random choice — same balance as the sequential
        # greedy node_ep_counts approach but per-update instead of globally.
        episodes_per_node = B // num_nodes
        remainder         = B % num_nodes
        start_nodes_arr   = np.repeat(np.arange(num_nodes), episodes_per_node)
        if remainder:
            extra = np.random.choice(num_nodes, size=remainder, replace=False)
            start_nodes_arr = np.concatenate([start_nodes_arr, extra])
        start_nodes_arr = np.random.permutation(start_nodes_arr).astype(np.int64)
        start_day_idxs  = np.random.randint(0, num_train_days, size=B).astype(np.int64)

        # Per-episode accumulators — each is a list of transition dicts.
        # Accumulated separately to keep episodes contiguous in the flat buffer,
        # since episodes of different lengths interleave in the step loop.
        ep_bufs        = [[] for _ in range(B)]
        last_values    = np.zeros(B, dtype=np.float32)
        truncated_flags = np.zeros(B, dtype=bool)

        masks  = vec_env.reset(start_nodes_arr, start_day_idxs)   # (B, N)
        active = np.ones(B, dtype=bool)

        with torch.no_grad():
            for step_i in range(num_nodes):   # defensive ceiling
                if not active.any():
                    break

                cur_nodes = vec_env.current_node     # (B,) int64
                time_el   = vec_env.time_elapsed     # (B,) float32
                step_cnt  = vec_env.step_count       # (B,) int64
                day_idx   = vec_env.current_day_idx  # (B,) int64

                # Feature construction — no Python loops over N or B
                node_feats = build_node_features_batch(
                    cur_nodes, start_nodes_arr, vec_env.visited_mask,
                    rm_pen_stack, time_matrix_arr, distance_arr,
                    day_idx, num_nodes, MAX_DURATION,
                    trucks_stack_raw=trucks_stack,
                    avail_prob_arr=avail_prob_arr,
                    reward_global_p95=reward_global_p95,
                )   # (B, N, 8)
                temporal = build_temporal_features_batch(time_el, step_cnt, MAX_DURATION, num_nodes)   # (B, 3)
                market   = build_market_features_batch(cur_nodes, day_idx, ltr_stack) if ltr_stack is not None \
                           else np.zeros((B, 1), dtype=np.float32)                                     # (B, 1)

                # Single forward pass over the full batch
                node_feats_t = torch.from_numpy(node_feats).to(DEVICE)
                temporal_t   = torch.from_numpy(temporal).to(DEVICE)
                market_t     = torch.from_numpy(market).to(DEVICE)
                cur_nodes_t  = torch.tensor(cur_nodes, dtype=torch.long, device=DEVICE)
                mask_t       = torch.from_numpy(masks).to(DEVICE)

                actions_t, log_probs_t, _, values_t = _forward(
                    agent, critic, node_feats_t, temporal_t, cur_nodes_t, mask_t,
                    market_feats_t=market_t,
                )
                actions   = actions_t.cpu().numpy()    # (B,) int64
                log_probs = log_probs_t.cpu().numpy()  # (B,)
                values    = values_t.cpu().numpy()     # (B,)

                # Vectorized environment step
                rewards, terminated, next_masks = vec_env.step(actions)

                # Write transitions — only for rows active BEFORE this step
                for b in np.where(active)[0]:
                    ep_bufs[b].append({
                        "node_feats":   node_feats[b],        # (N, 8)
                        "temporal":     temporal[b],           # (3,)
                        "market_feats": market[b],             # (1,)
                        "current_node": int(cur_nodes[b]),
                        "mask":         masks[b],              # (N,) int8
                        "action":       int(actions[b]),
                        "reward":       float(rewards[b]),
                        "log_prob":     float(log_probs[b]),
                        "value":        float(values[b]),
                        "done":         bool(terminated[b]),
                    })
                    if terminated[b]:
                        last_values[b] = 0.0   # MDP termination: no bootstrap

                active = active & ~terminated
                masks  = next_masks

        # Episodes still active after the step loop were truncated by the step ceiling
        if active.any():
            trunc_idx = np.where(active)[0]

            # Bootstrap value from a fresh critic forward pass over post-step state
            # (NOT recycling values[] from the last loop iteration — that was pre-step)
            cur_nodes_post = vec_env.current_node[trunc_idx]
            time_el_post   = vec_env.time_elapsed[trunc_idx]
            step_cnt_post  = vec_env.step_count[trunc_idx]
            day_idx_post   = vec_env.current_day_idx[trunc_idx]

            nf_trunc = build_node_features_batch(
                cur_nodes_post, start_nodes_arr[trunc_idx],
                vec_env.visited_mask[trunc_idx],
                rm_pen_stack, time_matrix_arr, distance_arr,
                day_idx_post, num_nodes, MAX_DURATION,
                trucks_stack_raw=trucks_stack,
                avail_prob_arr=avail_prob_arr,
                reward_global_p95=reward_global_p95,
            )
            tf_trunc = build_temporal_features_batch(time_el_post, step_cnt_post, MAX_DURATION, num_nodes)
            mf_trunc = build_market_features_batch(cur_nodes_post, day_idx_post, ltr_stack) if ltr_stack is not None \
                       else np.zeros((len(trunc_idx), 1), dtype=np.float32)

            with torch.no_grad():
                _, _, _, boot_values_t = _forward(
                    agent, critic,
                    torch.from_numpy(nf_trunc).to(DEVICE),
                    torch.from_numpy(tf_trunc).to(DEVICE),
                    torch.tensor(cur_nodes_post, dtype=torch.long, device=DEVICE),
                    torch.from_numpy(masks[trunc_idx]).to(DEVICE),
                    market_feats_t=torch.from_numpy(mf_trunc).to(DEVICE),
                )
            boot_values = boot_values_t.cpu().numpy()

            for i, b in enumerate(trunc_idx):
                truncated_flags[b] = True
                last_values[b]     = float(boot_values[i])

        # ── Self-critical baseline (Camino B): rollout greedy vectorizado ──────
        # Sobre los MISMOS start_nodes_arr/start_day_idxs del rollout muestreado,
        # para comparar en las mismas unidades y la misma realización del mundo.
        # Resetea vec_env (su estado post-rollout muestreado ya no se necesita:
        # el bootstrap de episodios truncados se calculó arriba).
        if USE_SELF_CRITICAL:
            greedy_ep_rewards = _collect_greedy_rollout(
                agent, vec_env, start_nodes_arr, start_day_idxs,
                rm_pen_stack, time_matrix_arr, distance_arr,
                trucks_stack, avail_prob_arr, reward_global_p95,
                ltr_stack, num_nodes,
            )
        else:
            greedy_ep_rewards = None

        # Flatten per-episode accumulators into the main RolloutBuffer
        # Episodes are written sequentially so compute_gae_per_episode sees
        # contiguous segments — its logic is unchanged.
        update_r_greedy = []
        for b in range(B):
            if not ep_bufs[b]:
                continue
            ep_start = len(buffer)
            for trans in ep_bufs[b]:
                buffer.add(
                    node_feats=trans["node_feats"],
                    temporal=trans["temporal"],
                    current_node=trans["current_node"],
                    mask=trans["mask"],
                    action=trans["action"],
                    reward=trans["reward"],
                    log_prob=trans["log_prob"],
                    value=trans["value"],
                    done=trans["done"],
                    market_feats=trans["market_feats"],
                )
            ep_end = len(buffer)
            buffer.episode_boundaries.append(
                (ep_start, ep_end, float(last_values[b]), bool(truncated_flags[b]))
            )
            ep_reward = sum(t["reward"] for t in ep_bufs[b])
            update_ep_rewards.append(ep_reward)
            episode_rewards.append(ep_reward)
            if USE_SELF_CRITICAL:
                update_r_greedy.append(float(greedy_ep_rewards[b]))

        buffer.compute_gae_per_episode(gamma=gamma, gae_lambda=gae_lambda)

        # ── Ventaja self-critical por episodio (Camino B) ──────────────────────
        sc_adv_raw_mean = 0.0
        r_greedy_avg    = 0.0
        if USE_SELF_CRITICAL:
            sc_advantage_raw = np.array(update_ep_rewards, dtype=np.float32) \
                              - np.array(update_r_greedy,  dtype=np.float32)
            sc_adv_raw_mean = float(sc_advantage_raw.mean())
            r_greedy_avg    = float(np.mean(update_r_greedy))

            sc_advantage_norm = (
                (sc_advantage_raw - sc_advantage_raw.mean())
                / (sc_advantage_raw.std() + 1e-8)
            )

            episode_idx_arr = np.zeros(len(buffer), dtype=np.int64)
            for k, (s, e, _, _) in enumerate(buffer.episode_boundaries):
                episode_idx_arr[s:e] = k

            buffer.sc_advantage = sc_advantage_norm
            buffer.episode_idx  = episode_idx_arr

        # ── Fase 2: Actualización PPO ─────────────────────────────────────────
        agent.train()
        critic.train()

        batch_total_losses  = []
        batch_actor_losses  = []
        batch_value_losses  = []
        batch_entropies     = []
        batch_kl_divs       = []
        batch_clip_fracs    = []
        batch_sc_losses     = []
        apply_self_critical = USE_SELF_CRITICAL and update >= SELF_CRITICAL_WARMUP_UPDATES

        for _ in range(n_ppo_epochs):

            kl_too_high = False # Flag para detectar si el KL se dispara en este epoch

            for _batch_idx, idx_batch in enumerate(buffer.get_batches(ppo_batch_size)):
                B = len(idx_batch)

                # Reconstruir tensores del batch
                nf_b   = torch.from_numpy(
                    np.stack([buffer.node_feats[i] for i in idx_batch])
                ).to(DEVICE)                                           # (B, N, N_NODE_FEATURES)
                tf_b   = torch.from_numpy(
                    np.stack([buffer.temporal_feats[i] for i in idx_batch])
                ).to(DEVICE)                                           # (B, 3)
                mf_b   = torch.from_numpy(
                    np.stack([buffer.market_feats[i] for i in idx_batch])
                ).to(DEVICE)                                           # (B, 1)
                cn_b   = torch.tensor(
                    [buffer.current_nodes[i] for i in idx_batch],
                    dtype=torch.long
                ).to(DEVICE)                                           # (B,)
                mask_b = torch.from_numpy(
                    np.stack([buffer.masks[i] for i in idx_batch])
                ).to(DEVICE)                                           # (B, N)
                act_b  = torch.tensor(
                    [buffer.actions[i] for i in idx_batch],
                    dtype=torch.long
                ).to(DEVICE)                                           # (B,)
                old_lp = torch.tensor(
                    [buffer.old_log_probs[i] for i in idx_batch],
                    dtype=torch.float32
                ).to(DEVICE)                                           # (B,)
                ret_b  = torch.tensor(
                    buffer.returns[idx_batch], dtype=torch.float32
                ).to(DEVICE)                                           # (B,)
                adv_b  = torch.tensor(
                    buffer.advantages[idx_batch], dtype=torch.float32
                ).to(DEVICE)                                           # (B,)

                if apply_self_critical:
                    sc_adv_b = torch.tensor(
                        buffer.sc_advantage[buffer.episode_idx[idx_batch]],
                        dtype=torch.float32,
                    ).to(DEVICE)                                       # (B,)

                # Normalización por mini-batch: cada gradiente ve ventajas
                # con media=0, std=1, independiente del resto del buffer.
                # Más estable que normalizar el buffer completo una sola vez
                # porque evita que outliers de otros mini-batches contaminen
                # la escala de la actualización actual.
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std(unbiased=False) + 1e-8)

                # Clipping de ventajas: limita casos extremos sin afectar
                # la dirección del gradiente (solo la magnitud).
                adv_b = torch.clamp(adv_b, -5.0, 5.0)

                check_tensor("advantages adv_b", adv_b)

                # Sentinelas para el bloque except (pueden no existir si el
                # error ocurre antes de que se calculen)
                new_lp = ratio = None

                try:
                    # Forward con gradientes
                    _, new_lp, entropy, new_val = _forward(
                        agent, critic, nf_b, tf_b, cn_b, mask_b,
                        actions_t=act_b, market_feats_t=mf_b,
                    )

                    # ── Loss del actor (PPO-Clip) ────────────────────────────
                    ratio       = torch.exp(new_lp - old_lp)
                    check_tensor("PPO ratio", ratio)

                    surr1       = ratio * adv_b
                    surr2       = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b
                    actor_loss  = -torch.min(surr1, surr2).mean()

                    # ── Loss del critic (MSE) ────────────────────────────────
                    value_loss  = nn.functional.mse_loss(new_val, ret_b)

                    # ── Bonus de entropía (exploración) ─────────────────────
                    entropy_loss = -entropy.mean()

                    actor_loss_total = actor_loss + entropy_coef * entropy_loss

                    # ── Self-critical (Camino B): término REINFORCE adicional ──
                    # No toca el critic — sc_loss depende solo de new_lp (actor).
                    if apply_self_critical:
                        sc_loss = -(SELF_CRITICAL_COEF * sc_adv_b * new_lp).mean()
                        actor_loss_total = actor_loss_total + sc_loss

                except RuntimeError as e:
                    sep = "=" * 60
                    print(f"\n{sep}")
                    print(f"NaN/Inf ABORT  update={update + 1}  batch={_batch_idx}")
                    print(f"  error        : {e}")
                    print(f"  avg reward   : {float(np.mean(update_ep_rewards)):.4f}")
                    print(f"  mask valid   : {mask_b.sum(dim=-1).tolist()}")
                    print(f"  action mask  :\n{mask_b.cpu().numpy()}")
                    print(
                        f"  adv_b        : min={adv_b.min():.4f}  max={adv_b.max():.4f}"
                        f"  mean={adv_b.mean():.4f}  std={adv_b.std():.4f}"
                    )
                    print(
                        f"  old_lp       : min={old_lp.min():.4f}  max={old_lp.max():.4f}"
                        f"  mean={old_lp.mean():.4f}"
                    )
                    if new_lp is not None:
                        print(
                            f"  new_lp       : min={new_lp.min():.4f}  max={new_lp.max():.4f}"
                            f"  mean={new_lp.mean():.4f}"
                        )
                    if ratio is not None:
                        print(
                            f"  ratio        : min={ratio.min():.4f}  max={ratio.max():.4f}"
                            f"  mean={ratio.mean():.4f}"
                        )
                    print(sep)
                    raise

                # ── Update actor & critic ─────────────────────────────────
                # value_loss también backpropaga hacia el encoder/context_net
                # del actor (h_t no se detacha antes del critic), así que un
                # solo backward sobre la suma evita recorrer ese tramo
                # compartido dos veces — los gradientes resultantes son
                # idénticos a hacer backward por separado (gradiente de la
                # suma = suma de gradientes), pero ~1.3x más rápido.
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                (actor_loss_total + value_loss).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
                actor_optimizer.step()
                nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                critic_optimizer.step()

                with torch.no_grad():
                    # KL aproximada: E[log π_old - log π_new]
                    approx_kl  = (old_lp - new_lp).mean().item()
                    # Fracción de ratios que el clip PPO recortó
                    clip_frac  = ((ratio - 1).abs() > clip_eps).float().mean().item()

                if approx_kl > 0.05:
                    kl_too_high = True                   # ← marca la bandera
                    batch_kl_divs.append(approx_kl)     # ← registra antes de salir
                    batch_clip_fracs.append(clip_frac)
                    break     
                
                batch_total_losses.append(actor_loss_total.item())
                batch_actor_losses.append(actor_loss.item())
                batch_value_losses.append(value_loss.item())
                batch_entropies.append(entropy.mean().item())
                batch_kl_divs.append(approx_kl)
                batch_clip_fracs.append(clip_frac)
                if apply_self_critical:
                    batch_sc_losses.append(sc_loss.item())

            if kl_too_high:  # sale también de n_ppo_epochs
                break

        # Explained variance: cuánto explica el critic los retornos reales
        ret_all = buffer.returns
        val_all = np.array(buffer.old_values, dtype=np.float32)
        var_ret = np.var(ret_all)
        explained_var = (
            float(1.0 - np.var(ret_all - val_all) / (var_ret + 1e-8))
            if var_ret > 1e-8 else 0.0
        )

        avg_reward    = float(np.mean(update_ep_rewards))
        avg_loss      = float(np.mean(batch_total_losses))  if batch_total_losses  else 0.0
        avg_pol_loss  = float(np.mean(batch_actor_losses))  if batch_actor_losses  else 0.0
        avg_val_loss  = float(np.mean(batch_value_losses))  if batch_value_losses  else 0.0
        avg_entropy   = float(np.mean(batch_entropies))     if batch_entropies     else 0.0
        avg_kl        = float(np.mean(batch_kl_divs))       if batch_kl_divs       else 0.0
        avg_clip_frac = float(np.mean(batch_clip_fracs))    if batch_clip_fracs    else 0.0
        avg_sc_loss   = float(np.mean(batch_sc_losses))     if batch_sc_losses     else 0.0

        episode_losses.append(avg_loss)
        log_entry = {
            "update":           update + 1,
            "reward":           avg_reward,
            "total_loss":       avg_loss,
            "policy_loss":      avg_pol_loss,
            "value_loss":       avg_val_loss,
            "entropy":          avg_entropy,
            "kl_divergence":    avg_kl,
            "clip_fraction":    avg_clip_frac,
            "explained_var":    explained_var,
            "entropy_coef":     entropy_coef,
            "actor_lr":         actor_lr,
        }
        if USE_SELF_CRITICAL:
            log_entry["r_greedy_avg"]      = r_greedy_avg
            log_entry["sc_advantage_mean"] = sc_adv_raw_mean
            log_entry["sc_loss_value"]     = avg_sc_loss

        log_freq = max(1, n_updates // 20)
        if (update + 1) % log_freq == 0:
            greedy_reward, greedy_valid_rate = _evaluate_greedy(
                agent, eval_start_nodes, eval_start_days,
                time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
                ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95,
            )
            log_entry["greedy_reward"]     = greedy_reward
            log_entry["greedy_valid_rate"] = greedy_valid_rate

            episodes_done = (update + 1) * n_episodes_per_update
            sc_log_suffix = (
                f" | r_greedy {r_greedy_avg:6.1f}"
                f" | sc_adv {sc_adv_raw_mean:6.2f}"
                f" | sc_loss {avg_sc_loss:.4f}"
            ) if USE_SELF_CRITICAL else ""
            print(
                f"  update {update+1:>5}/{n_updates} "
                f"| ep {episodes_done:>6}/{total_episodes} "
                f"| reward {avg_reward:6.1f} "
                f"| greedy {greedy_reward:6.1f} "
                f"| greedy_valid {greedy_valid_rate:.2f} "
                f"| loss {avg_loss:.4f} "
                f"| pol {avg_pol_loss:.4f} "
                f"| val {avg_val_loss:.4f} "
                f"| ent {avg_entropy:.3f} "
                f"| kl {avg_kl:.4f} "
                f"| clip {avg_clip_frac:.2f} "
                f"| ev {explained_var:.3f}"
                f"{sc_log_suffix}"
            )

        training_log.append(log_entry)

    print("Entrenamiento AM-PPO completo.")
    return agent, critic, episode_rewards, episode_losses, training_log
