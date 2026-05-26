"""
state_features.py
=================
State-vector construction for the AM-PPO pipeline.

State layout (length = 2 + num_nodes + 2):
  [0]           current node index
  [1]           time elapsed / max_duration  (normalised 0-1)
  [2..2+N-1]    one-hot visited encoding     (1 = visited)
  [2+N]         remaining_nodes_fraction     = (num_nodes - len(visited_set)) / num_nodes
  [2+N+1]       progress_fraction            = step / num_nodes
"""

import numpy as np


def get_state_size(num_nodes: int) -> int:
    return 2 + num_nodes + 2


def build_state(
    current_node: int,
    time_elapsed: float,
    visited_set: set,
    step: int,
    max_duration: float,
    num_nodes: int,
) -> np.ndarray:
    """Return a float32 state vector of length get_state_size(num_nodes)."""
    state = np.zeros(get_state_size(num_nodes), dtype=np.float32)
    state[0] = current_node
    state[1] = min(time_elapsed, max_duration) / max_duration
    for v in visited_set:
        if 0 <= v < num_nodes:
            state[2 + v] = 1.0
    state[2 + num_nodes]     = (num_nodes - len(visited_set)) / num_nodes
    state[2 + num_nodes + 1] = step / max(num_nodes, 1)
    return state
