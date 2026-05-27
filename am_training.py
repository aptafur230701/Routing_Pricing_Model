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
        noise_sigma, num_nodes
    )

· Usa RoutingEnv como entorno estándar (Siguiendo practicas de API Gymnasium).
"""

import random
import numpy as np
import torch
import torch.nn as nn

from config import (
    MAX_DURATION,
    DEVICE,
    SEED,
    get_episodes_per_node,
    AM_D_H, AM_N_HEADS, AM_N_LAYERS, AM_D_FF,
    PPO_N_EPISODES_PER_UPDATE, PPO_N_EPOCHS, PPO_BATCH_SIZE,
    PPO_LR, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_CLIP_EPS,
    PPO_ENTROPY_COEF, PPO_GRAD_CLIP,
)
from routing_env import RoutingEnv
from problem_data import build_day_matrices
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
        self.node_feats    = []   # list of np.ndarray (N, 5)
        self.temporal_feats = []  # list of np.ndarray (3,)
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
    ):
        self.node_feats.append(node_feats)
        self.temporal_feats.append(temporal)
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
    agent:        AMRoutingAgent,
    critic:       CriticHead,
    node_feats_t: torch.Tensor,    # (B, N, 5)
    temporal_t:   torch.Tensor,    # (B, 3)
    current_nodes: torch.Tensor,   # (B,) int64
    mask_t:       torch.Tensor,    # (B, N) int8
    actions_t:    torch.Tensor = None,  # (B,) int64 — None en recolección
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

    h_t = agent.context_net(graph_emb, current_emb, temporal_t)   # (B, d_h)
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
# Función principal de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def run_am_training(
    time_matrix,
    rate_stack:   np.ndarray,
    loads_stack:  np.ndarray,
    distance_arr: np.ndarray,
    diesel_arr:   np.ndarray,
    noise_sigma:  float,
    num_nodes:    int,
    pretrained_agent:  AMRoutingAgent = None,
    pretrained_critic: CriticHead     = None,
) -> tuple:
    """
    Entrena AMRoutingAgent + CriticHead con PPO.

    Parámetros
    ----------
    time_matrix   : pd.DataFrame — tiempos entre nodos.
    rate_stack    : np.ndarray [num_days, N, N]
    loads_stack   : np.ndarray [num_days, N, N]
    distance_arr  : np.ndarray [N, N]
    diesel_arr    : np.ndarray [N, N]
    noise_sigma   : float
    num_nodes     : int

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
    entropy_coef          = PPO_ENTROPY_COEF
    grad_clip             = PPO_GRAD_CLIP

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    episodes_per_node = get_episodes_per_node(num_nodes)
    total_episodes    = episodes_per_node * num_nodes
    n_updates         = max(1, total_episodes // n_episodes_per_update)
    num_days          = rate_stack.shape[0]

    # ── Modelos ───────────────────────────────────────────────────────────────
    agent  = pretrained_agent  if pretrained_agent  is not None \
             else AMRoutingAgent(num_nodes, d_h, n_heads, n_layers, d_ff, device=DEVICE)
    critic = pretrained_critic if pretrained_critic is not None \
             else CriticHead(d_h).to(DEVICE)

    # Optimizers separados: el critic necesita converger más rápido que el actor
    # para dar señales de ventaja de calidad. Con un optimizer compartido y lr=3e-5
    # el critic aprende demasiado lento y el EV se estanca por debajo de 0.7.
    # Ratio 10x entre critic y actor es estándar en PPO para problemas combinatorios.
    actor_optimizer  = torch.optim.Adam(agent.parameters(),  lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr * 10)

    # ── Entorno ───────────────────────────────────────────────────────────────
    _, rm_init = build_day_matrices(
        rate_stack[0], loads_stack[0], distance_arr, diesel_arr
    )
    env = RoutingEnv(
        time_matrix=time_matrix,
        reward_matrix_penalized=rm_init,
        noise_sigma=noise_sigma,
        num_nodes=num_nodes,
        max_duration=MAX_DURATION,
    )

    print(
        f"\n--- Entrenamiento AM-PPO: {total_episodes} episodios "
        f"({n_updates} updates × {n_episodes_per_update} ep/update) ---"
    )

    episode_rewards = []
    episode_losses  = []
    training_log    = []
    node_ep_counts  = {n: 0 for n in range(num_nodes)}

    # ── Loop principal ────────────────────────────────────────────────────────
    for update in range(n_updates):
        buffer = RolloutBuffer()
        update_ep_rewards = []

        # ── Fase 1: Recolección de rollouts ───────────────────────────────────
        agent.eval()
        critic.eval()

        for _ in range(n_episodes_per_update):
            # Selección balanceada del nodo de inicio
            min_count = min(node_ep_counts.values())
            candidates = [n for n, c in node_ep_counts.items() if c == min_count]
            start_node = random.choice(candidates)
            node_ep_counts[start_node] += 1

            # Samplear día aleatorio
            day_idx = np.random.randint(0, num_days)
            _, rm_pen = build_day_matrices(
                rate_stack[day_idx], loads_stack[day_idx], distance_arr, diesel_arr
            )
            env.update_reward_matrix(rm_pen)
            obs, info = env.reset(options={"start_node": start_node})

            ep_start_idx = len(buffer)
            ep_reward    = 0.0
            last_value   = 0.0
            ep_truncated = False

            with torch.no_grad():
                for step in range(num_nodes):  # defensive ceiling; env terminates naturally
                    mask = info["action_mask"]   # (N,) int8
                    current_node_before_step = env.current_node

                    action, log_prob, _, value, nf, tf = agent.act_with_value(
                        critic,
                        env.current_node, env.start_node, env.visited_set,
                        env.time_elapsed, step,
                        rm_pen, time_matrix, distance_arr,
                        MAX_DURATION,
                        action_mask=mask,
                    )

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    if truncated:
                        ep_truncated = True
                        # ANTES: reward += INCOMPLETE_PENALTY
                        # Esto sumaba el penalty al reward que va al buffer (línea 354)
                        # y también a ep_reward (línea 360), contándolo dos veces.
                        # AHORA: solo se registra en ep_reward como señal de diagnóstico,
                        # pero NO entra al buffer — GAE lo maneja vía last_value.
                        last_value = agent.estimate_value(
                            critic,
                            env.current_node, env.start_node, env.visited_set,
                            env.time_elapsed, step + 1,
                            rm_pen, time_matrix, distance_arr,
                            MAX_DURATION,
                        )

                    buffer.add(
                        node_feats=nf,
                        temporal=tf,
                        current_node=current_node_before_step,
                        mask=mask,
                        action=action,
                        reward=float(reward),   # reward limpio, sin penalty artificial
                        log_prob=log_prob.item(),
                        value=value,
                        done=done,
                    )

                    ep_reward += reward
                    obs = next_obs

                    if done:
                        break

            buffer.episode_boundaries.append(
                (ep_start_idx, len(buffer), last_value, ep_truncated)
            )
            update_ep_rewards.append(ep_reward)
            episode_rewards.append(ep_reward)

        buffer.compute_gae_per_episode(gamma=gamma, gae_lambda=gae_lambda)

        # ── Fase 2: Actualización PPO ─────────────────────────────────────────
        agent.train()
        critic.train()

        batch_total_losses  = []
        batch_actor_losses  = []
        batch_value_losses  = []
        batch_entropies     = []
        batch_kl_divs       = []
        batch_clip_fracs    = []

        for _ in range(n_ppo_epochs):

            kl_too_high = False # Flag para detectar si el KL se dispara en este epoch

            for _batch_idx, idx_batch in enumerate(buffer.get_batches(ppo_batch_size)):
                B = len(idx_batch)

                # Reconstruir tensores del batch
                nf_b   = torch.from_numpy(
                    np.stack([buffer.node_feats[i] for i in idx_batch])
                ).to(DEVICE)                                           # (B, N, 5)
                tf_b   = torch.from_numpy(
                    np.stack([buffer.temporal_feats[i] for i in idx_batch])
                ).to(DEVICE)                                           # (B, 3)
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
                        agent, critic, nf_b, tf_b, cn_b, mask_b, actions_t=act_b
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
                # Ambos .backward() deben ejecutarse antes de cualquier
                # .step() para evitar que las actualizaciones in-place del
                # optimizador invaliden el grafo compartido del forward pass.
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                actor_loss_total.backward(retain_graph=True)
                value_loss.backward()
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

        episode_losses.append(avg_loss)
        training_log.append({
            "update":           update + 1,
            "reward":           avg_reward,
            "total_loss":       avg_loss,
            "policy_loss":      avg_pol_loss,
            "value_loss":       avg_val_loss,
            "entropy":          avg_entropy,
            "kl_divergence":    avg_kl,
            "clip_fraction":    avg_clip_frac,
            "explained_var":    explained_var,
        })

        log_freq = max(1, n_updates // 20)
        if (update + 1) % log_freq == 0:
            episodes_done = (update + 1) * n_episodes_per_update
            print(
                f"  update {update+1:>5}/{n_updates} "
                f"| ep {episodes_done:>6}/{total_episodes} "
                f"| reward {avg_reward:6.1f} "
                f"| loss {avg_loss:.4f} "
                f"| pol {avg_pol_loss:.4f} "
                f"| val {avg_val_loss:.4f} "
                f"| ent {avg_entropy:.3f} "
                f"| kl {avg_kl:.4f} "
                f"| clip {avg_clip_frac:.2f} "
                f"| ev {explained_var:.3f}"
            )

    print("Entrenamiento AM-PPO completo.")
    return agent, critic, episode_rewards, episode_losses, training_log
