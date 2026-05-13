"""
state.py
========
State-vector construction for the DRL agent.

State layout (length = 2 + num_nodes + 2):
  [0]           current node index
  [1]           time elapsed / max_duration  (normalised 0-1)
  [2..2+N-1]    one-hot visited encoding     (1 = visited)
  [2+N]         remaining steps / max_steps  (normalised 0-1)
  [2+N+1]       current step / max_steps     (normalised 0-1)
"""

import numpy as np


def get_state_size(num_nodes: int) -> int:
    return 2 + num_nodes + 2


def build_state(
    current_node:  int,
    time_elapsed:  float,
    visited_set:   set,
    step:          int,
    max_duration:  float,
    max_steps:     int,
    num_nodes:     int,
) -> np.ndarray:
    """Return a float32 state vector of length get_state_size(num_nodes)."""
    state = np.zeros(get_state_size(num_nodes), dtype=np.float32)

    state[0] = current_node
    state[1] = min(time_elapsed, max_duration) / max_duration

    for v in visited_set:
        if 0 <= v < num_nodes:
            state[2 + v] = 1.0

    state[2 + num_nodes]     = (max_steps - step) / max_steps   # remaining
    state[2 + num_nodes + 1] = step / max_steps                  # progress

    return state
