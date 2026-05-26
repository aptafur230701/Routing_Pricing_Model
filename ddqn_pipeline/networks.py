"""
networks.py
===========
Neural network architectures for the DRL agent.

  DuelingQNetwork  — Dueling DQN (Wang et al., 2016)
                     Shared trunk → V(s) stream + A(s,a) stream → Q(s,a)
  ValueNetwork     — Lightweight V(s) baseline for variance reduction
                     under stochastic rewards.
"""

import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    """Dueling DQN: separates state value V(s) from advantage A(s,a).

    Q(s, a) = V(s) + A(s, a) - mean_a'[ A(s, a') ]

    Subtracting the mean advantage makes the decomposition unique and
    prevents V and A from drifting to arbitrary offset solutions.
    """

    def __init__(self, state_size: int, action_size: int,
                 h1: int, h2: int, h3: int, h4: int):
        super().__init__()

        # Shared trunk
        self.fc1 = nn.Linear(state_size, h1);  self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2);          self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3);          self.bn3 = nn.BatchNorm1d(h3)

        # Value stream  → scalar V(s)
        self.fc_v1 = nn.Linear(h3, h4)
        self.fc_v2 = nn.Linear(h4, 1)

        # Advantage stream → A(s, a) for each action
        self.fc_a1 = nn.Linear(h3, h4)
        self.fc_a2 = nn.Linear(h4, action_size)

        # Xavier initialisation for all linear layers
        for layer in [self.fc1, self.fc2, self.fc3,
                      self.fc_v1, self.fc_v2, self.fc_a1, self.fc_a2]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))

        v = F.relu(self.fc_v1(x));  v = self.fc_v2(v)
        a = F.relu(self.fc_a1(x));  a = self.fc_a2(a)

        return v + (a - a.mean(dim=1, keepdim=True))


class ValueNetwork(nn.Module):
    """State value function V(s).

    Trained to approximate E[Q(s, a*)] and used as a variance-reduction
    baseline when rewards are stochastic.  Smaller than DuelingQNetwork
    by design — it only needs to capture coarse state quality.
    """

    def __init__(self, state_size: int, num_nodes: int):
        super().__init__()
        h = num_nodes * num_nodes          # e.g. 100 for 10 nodes

        self.fc1 = nn.Linear(state_size, h);  self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h);           self.bn2 = nn.BatchNorm1d(h)
        self.fc3 = nn.Linear(h, 1)            # linear output — V(s) can be negative

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


def get_default_network_sizes(num_nodes: int):
    """Heuristic layer sizes scaled to the graph dimension."""
    n2 = num_nodes * num_nodes
    return n2, n2 * 2, n2 * 2, n2 // 2    # h1, h2, h3, h4
