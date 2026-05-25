"""
config.py
=========
All global constants, toggles, and hyperparameter defaults.
Nothing is imported from other project modules here.
"""

import torch

# ── Problem identity ──────────────────────────────────────────
MPG             = 6.5

# ── Stochastic reward ────────────────────────────────────────
STOCHASTIC_MODE  = True
NOISE_FRACTION   = 0.10          # 10 % of reward-matrix std

# ── Optimisation toggles (set False for ablation) ────────────
DOUBLE_DQN        = True
PER_ENABLED       = True
PER_ALPHA         = 0.6
PER_BETA_START    = 0.4
PER_BETA_END      = 1.0
PER_EPSILON       = 1e-5
GRAD_CLIP_NORM    = 1.0
LR_SCHEDULE       = "cosine"     # "cosine" | "constant"
EPSILON_SCHEDULE  = "cosine"     # "cosine" | "linear"

# ── Environment constraints ───────────────────────────────────
DURATION_LIMIT      = 70.0
DURATION_TOLERANCE  = 0.10
MIN_DURATION        = DURATION_LIMIT * (1 - DURATION_TOLERANCE)
MAX_DURATION        = DURATION_LIMIT * (1 + DURATION_TOLERANCE)

# ── Reward shaping ────────────────────────────────────────────
REWARD_SCALE_FACTOR    = 100.0
RETURN_SUCCESS_BONUS   =  500.0 / REWARD_SCALE_FACTOR
TIME_VIOLATION_PENALTY =  -3000.0 / REWARD_SCALE_FACTOR
INCOMPLETE_PENALTY     =  -2000.0 / REWARD_SCALE_FACTOR
BIG_M_PENALTY          = -1e9

# ── Default training hyperparameters (overridden by Optuna) ──
LEARNING_RATE        = 0.0001
GAMMA                = 0.95
EPSILON_START        = 1.0
EPSILON_END          = 0.05
BATCH_SIZE           = 32
TARGET_UPDATE_FREQ   = 50

# ── Evaluation ────────────────────────────────────────────────
N_EVAL_EPISODES = 50

# ── Reproducibility ───────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_episodes_per_node(num_nodes: int) -> int:
    """Episodes per start-node for the full training run."""
    if num_nodes <= 10:  return 5000
    if num_nodes <= 15:  return 5500
    if num_nodes <= 20:  return 6000
    return 7000


def get_buffer_size(num_nodes: int) -> int:
    return max(20_000, num_nodes * num_nodes * 40)
