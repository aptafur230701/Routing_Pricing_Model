"""
config.py
=========
All global constants, toggles, and hyperparameter defaults for the AM-PPO pipeline.
This file is imported by all modules, so it should not contain any heavy dependencies.

"""

import torch

# ── Problem identity ──────────────────────────────────────────
MPG             = 6.5
MARGINAL_COST_SIN_DIESEL = 1.73

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

# ── Evaluation ────────────────────────────────────────────────
N_EVAL_EPISODES = 50
N_DRL_REAL_SAMPLES = 5   # número de rollouts estocásticos en inferencia

# ── AM Model architecture ─────────────────────────────────────
AM_D_H      = 128  # Dimensión de embeddings del Transformer
AM_N_HEADS  = 8    # Número de cabezas de atención en el Transformer
AM_N_LAYERS = 3    # Número de capas del encoder Transformer
AM_D_FF     = 512  # Dimensión de la capa feed-forward

# ── PPO hyperparameters ───────────────────────────────────────
PPO_N_EPISODES_PER_UPDATE = 360    # Episodios recolectados antes de cada update PPO
PPO_N_EPOCHS              = 4      # Épocas de entrenamiento PPO por rollout
PPO_BATCH_SIZE            = 64     # Tamaño de batch para entrenamiento PPO
PPO_LR                    = 1e-5   # Learning rate para el actor
PPO_GAMMA                 = 0.99   # Factor de descuento para las recompensas futuras
PPO_GAE_LAMBDA            = 0.95   # Factor de GAE
PPO_CLIP_EPS              = 0.15   # Clip PPO para limitar cambios de política
PPO_ENTROPY_COEF          = 0.05   # Coeficiente de entropía para PPO
PPO_GRAD_CLIP             = 0.5    # Clipping de gradiente para PPO

# ── Train / eval split ───────────────────────────────────────
TRAIN_DAYS = 90   # días usados para entrenamiento; el resto (días 90-119) queda para evaluación

# ── Market signal normalisation clamps ───────────────────────
LTR_CLIP    = 200.0   # p95 del LTR es 124; 200 es techo seguro
TRUCKS_CLIP =  50.0   # p95 de trucks delta=1 es 45

# ── Reproducibility ───────────────────────────────────────────
SEED = 42

# ── Device ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── MIP exact baseline ────────────────────────────────────
MIP_FLOOR_EPS    = 1e-6       # ε para la desigualdad estricta del floor de bucket
                               # (desplaza el límite superior del bucket un ε hacia
                               #  abajo para que T_k en el borde exacto de un múltiplo
                               #  de DAYS_PER_PERIOD caiga en el bucket correcto)

def get_mip_time_limit(num_nodes: int) -> int:
    """Tiempo límite HiGHS por instancia, calibrado experimentalmente.

    HiGHS necesita ~120s de presolving para N=20 antes de encontrar la primera
    solución factible; con 300s llega al ~97% del óptimo en la mayoría de instancias.
    N=10 converge en <30s con HiGHS, pero se conserva 120s como margen.
    """
    if num_nodes <= 10:  return 120   # HiGHS cierra en <30s; 120s es margen amplio
    if num_nodes <= 20:  return 420   # HiGHS necesita ~365s para probar optimalidad; 420s da margen
    if num_nodes <= 35:  return 120
    return 60


def get_episodes_per_node(num_nodes: int) -> int:
    """Episodes per start-node for the full training run."""
    if num_nodes <= 10:  return 5000
    if num_nodes <= 20:  return 9000
    if num_nodes <= 35:  return 6500
    if num_nodes <= 50:  return 7000
    if num_nodes <= 75:  return 7500
    return 8000          # 100 nodos


def get_beam_width(num_nodes: int) -> int:
    """Beam width for inference, scaled by problem size.
    Modelos pequeños necesitan más exploración en inferencia.
    Modelos grandes tienen políticas más robustas y beam=1 es suficiente."""
    if num_nodes <= 10: return 5   # modelo pequeño, necesita más exploración
    if num_nodes <= 20: return 5   # subido de 3 a 5
    if num_nodes <= 35: return 3   # balance costo/calidad
    return 1                       # modelo grande, confiar en la política

