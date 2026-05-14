"""
agent.py
========
DQNAgent_Optimized: the full DRL agent combining
  · Dueling DQN architecture
  · Double DQN target computation
  · Prioritized Experience Replay (PER)
  · Cosine-annealed epsilon and learning rate
  · Gradient clipping for stable stochastic training
"""

import os
import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DOUBLE_DQN, PER_ENABLED, PER_ALPHA, PER_BETA_START, PER_BETA_END,
    GRAD_CLIP_NORM, LR_SCHEDULE, EPSILON_SCHEDULE,
)
from networks import DuelingQNetwork, ValueNetwork, get_default_network_sizes
from replay_buffer import PrioritizedReplayBuffer


class DQNAgent_Optimized:

    def __init__(
        self,
        state_size:          int,
        action_size:         int,
        learning_rate:       float,
        gamma:               float,
        buffer_size:         int,
        batch_size:          int,
        device,
        num_nodes:           int,
        total_training_steps: int,
        epsilon_start:       float = 1.0,
        epsilon_end:         float = 0.05,
        epsilon_decay_steps: int   = None,
        target_update_freq:  int   = 50,
        h1: int = None, h2: int = None, h3: int = None, h4: int = None,
        per_alpha:           float = PER_ALPHA,
        per_beta_start:      float = PER_BETA_START,
        grad_clip:           float = GRAD_CLIP_NORM,
    ):
        self.state_size           = state_size
        self.action_size          = action_size
        self.gamma                = gamma
        self.batch_size           = batch_size
        self.device               = device
        self.num_nodes            = num_nodes
        self.total_training_steps = max(1, total_training_steps)
        self.epsilon_start        = epsilon_start
        self.epsilon_end          = epsilon_end
        self.epsilon_decay_steps  = epsilon_decay_steps or total_training_steps
        self.target_update_freq   = target_update_freq
        self.grad_clip            = grad_clip
        self.per_beta_start       = per_beta_start
        self.epsilon              = epsilon_start

        # Network sizes
        dh1, dh2, dh3, dh4 = get_default_network_sizes(num_nodes)
        h1 = h1 or dh1;  h2 = h2 or dh2
        h3 = h3 or dh3;  h4 = h4 or dh4

        # Q-Networks (policy + frozen target)
        self.policy_net = DuelingQNetwork(state_size, action_size, h1, h2, h3, h4).to(device)
        self.target_net = DuelingQNetwork(state_size, action_size, h1, h2, h3, h4).to(device)
        self.optimizer      = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_function  = nn.MSELoss(reduction='none')

        # V-Networks (value baseline)
        self.value_net        = ValueNetwork(state_size, num_nodes).to(device)
        self.value_target_net = ValueNetwork(state_size, num_nodes).to(device)
        self.value_optimizer      = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.value_loss_function  = nn.MSELoss(reduction='none')

        # Replay buffer
        if PER_ENABLED:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        else:
            self.memory = deque(maxlen=buffer_size)

        # LR schedulers
        if LR_SCHEDULE == "cosine":
            T = max(1, total_training_steps)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T, eta_min=learning_rate * 0.01)
            self.value_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.value_optimizer, T_max=T, eta_min=learning_rate * 0.01)
        else:
            self.scheduler = self.value_scheduler = None

        self.update_target_model()
        self.target_net.eval()
        self.value_target_net.eval()

    # ── target sync ──────────────────────────────────────────
    def update_target_model(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.value_target_net.load_state_dict(self.value_net.state_dict())

    # ── memory ───────────────────────────────────────────────
    def remember(self, state, action, reward, next_state, done):
        if PER_ENABLED:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))

    # ── action selection ─────────────────────────────────────
    def act(self, state, invalid_actions: set = None):
        """Epsilon-greedy with action masking (never revisit, never self-loop)."""
        if invalid_actions is None:
            invalid_actions = set()

        current_node  = int(state[0])
        blocked       = invalid_actions | {current_node}
        valid_actions = [a for a in range(self.action_size) if a not in blocked]

        if not valid_actions:
            return current_node

        if random.random() <= self.epsilon:
            return random.choice(valid_actions)

        self.policy_net.eval()
        with torch.no_grad():
            q = self.policy_net(
                torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            ).cpu().numpy()[0]
        self.policy_net.train()

        for b in blocked:
            if 0 <= b < self.action_size:
                q[b] = -np.inf

        best = int(np.argmax(q))
        if q[best] == -np.inf:
            return random.choice(valid_actions)
        return best

    # ── buffer diagnostics ───────────────────────────────────
    def get_buffer_stats(self, current_step: int) -> tuple:
        """Returns (buffer_size, current_beta, mean_leaf_priority)."""
        size = len(self.memory)
        beta = min(1.0, self.per_beta_start + (PER_BETA_END - self.per_beta_start) * (
            current_step / max(1, self.total_training_steps)))
        mean_priority = 0.0
        if PER_ENABLED and hasattr(self.memory, 'tree'):
            tree = self.memory.tree
            if tree.n_entries > 0:
                leaf_start = tree.capacity - 1
                leaf_prio  = tree.tree[leaf_start:leaf_start + tree.n_entries]
                nz = leaf_prio[leaf_prio > 0]
                if len(nz) > 0:
                    mean_priority = float(nz.mean())
        return size, float(beta), mean_priority

    # ── learning step ────────────────────────────────────────
    def replay(self, current_step: int = 0) -> tuple:
        """Sample a mini-batch and update both Q- and V-networks.
        Returns (q_loss, v_loss, grad_norm). All zero when buffer not full."""
        if len(self.memory) < self.batch_size:
            return 0.0, 0.0, 0.0

        # Sample
        if PER_ENABLED:
            beta = self.per_beta_start + (PER_BETA_END - self.per_beta_start) * (
                current_step / max(1, self.total_training_steps))
            indices, minibatch, is_weights = self.memory.sample(self.batch_size, beta)
            if len(minibatch) < self.batch_size // 2:
                return 0.0, 0.0, 0.0
            is_w = torch.from_numpy(is_weights).float().to(self.device).unsqueeze(1)
        else:
            minibatch = random.sample(self.memory, self.batch_size)
            is_w = torch.ones(len(minibatch), 1).to(self.device)

        states      = torch.from_numpy(np.vstack([e[0] for e in minibatch])).float().to(self.device)
        actions     = torch.from_numpy(np.vstack([e[1] for e in minibatch])).long().to(self.device)
        rewards     = torch.from_numpy(np.vstack([e[2] for e in minibatch])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch])).float().to(self.device)
        dones       = torch.from_numpy(np.vstack([e[4] for e in minibatch]).astype(np.uint8)).float().to(self.device)

        # ── Q-Network update (Double DQN) ────────────────────
        with torch.no_grad():
            if DOUBLE_DQN:
                best_a  = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                q_next  = self.target_net(next_states).gather(1, best_a)
            else:
                q_next  = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            q_targets = rewards + self.gamma * q_next * (1 - dones)

        q_pred    = self.policy_net(states).gather(1, actions)
        q_loss    = (self.loss_function(q_pred, q_targets) * is_w).mean()

        self.optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # ── V-Network update ─────────────────────────────────
        with torch.no_grad():
            if DOUBLE_DQN:
                best_av = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                q_next_v = self.target_net(next_states).gather(1, best_av)
            else:
                q_next_v = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            v_targets = rewards + self.gamma * q_next_v * (1 - dones)

        v_pred  = self.value_net(states)
        v_loss  = (self.value_loss_function(v_pred, v_targets) * is_w).mean()

        self.value_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.grad_clip)
        self.value_optimizer.step()

        # ── PER priority update ──────────────────────────────
        if PER_ENABLED:
            td_errors = (q_pred - q_targets).detach().cpu().numpy().flatten()
            self.memory.update_priorities(indices, td_errors)

        if self.scheduler:        self.scheduler.step()
        if self.value_scheduler:  self.value_scheduler.step()

        return q_loss.item(), v_loss.item(), q_grad_norm.item()

    # ── epsilon decay ────────────────────────────────────────
    def decay_epsilon(self, current_step: int):
        ratio = current_step / max(1, self.epsilon_decay_steps)
        if EPSILON_SCHEDULE == "cosine":
            self.epsilon = self.epsilon_end + 0.5 * (self.epsilon_start - self.epsilon_end) * (
                1 + math.cos(math.pi * ratio))
        else:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon_start - (self.epsilon_start - self.epsilon_end) * ratio)

    # ── persistence ──────────────────────────────────────────
    def save(self, path: str):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'policy_net':       self.policy_net.state_dict(),
                'value_net':        self.value_net.state_dict(),
                'optimizer':        self.optimizer.state_dict(),
                'value_optimizer':  self.value_optimizer.state_dict(),
            }, path)
            print(f"Model saved → {path}")
        except Exception as e:
            print(f"Save error: {e}")

    def load(self, path: str):
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.value_net.load_state_dict(ckpt['value_net'])
            self.update_target_model()
            print(f"Model loaded ← {path}")
        except Exception as e:
            print(f"Load error: {e}")
