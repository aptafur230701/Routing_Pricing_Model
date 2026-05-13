"""
replay_buffer.py
================
Prioritized Experience Replay (PER) implementation.
  · SumTree  — O(log n) weighted sampling
  · PrioritizedReplayBuffer — stores transitions, samples by TD-error priority
"""

import random
import numpy as np
from config import PER_ALPHA, PER_EPSILON


class SumTree:
    """Binary tree where every parent = sum of its children.
    Enables O(log n) priority-proportional sampling."""

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1)
        self.data      = [None] * capacity
        self.write_idx = 0
        self.n_entries = 0

    # ── internal helpers ──────────────────────────────────────
    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # ── public API ────────────────────────────────────────────
    def total(self) -> float:
        return self.tree[0]

    def add(self, priority: float, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float):
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Experience replay with proportional prioritization (Schaul et al., 2016).

    Transitions with higher TD-error are sampled more often.
    Importance-sampling (IS) weights correct for the introduced bias.
    """

    def __init__(self, capacity: int, alpha: float = PER_ALPHA):
        self.tree         = SumTree(capacity)
        self.alpha        = alpha
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        priority   = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int, beta: float = 0.4):
        indices    = []
        batch      = []
        is_weights = []
        segment    = self.tree.total() / batch_size

        # min_prob from actual non-zero leaf priorities (avoids empty-slot zeros)
        leaf_start       = self.tree.capacity - 1
        leaf_priorities  = self.tree.tree[leaf_start:leaf_start + self.tree.n_entries]
        nonzero          = leaf_priorities[leaf_priorities > 0]
        min_priority     = nonzero.min() if len(nonzero) > 0 else PER_EPSILON
        min_prob         = (min_priority / self.tree.total()
                            if self.tree.total() > 0 else PER_EPSILON)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)

            if data is None:          # fallback resample
                s = random.uniform(0, self.tree.total() - 1e-6)
                idx, priority, data = self.tree.get(s)
            if data is None:
                continue

            prob      = priority / self.tree.total()
            is_weight = (self.tree.n_entries * prob) ** (-beta)
            indices.append(idx)
            batch.append(data)
            is_weights.append(is_weight)

        if is_weights:
            max_w      = max(is_weights)
            is_weights = [w / max_w for w in is_weights]

        return indices, batch, np.array(is_weights, dtype=np.float32)

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            priority          = (abs(td_error) + PER_EPSILON) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries
