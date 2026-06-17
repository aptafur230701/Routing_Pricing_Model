"""
test_smoke_training.py
======================
Smoke test: monkey-patches PPO config to do only 3 updates and runs the full
training pipeline on N=10 to confirm no NaN/Inf and report timing.

Usage:
    python3 test_smoke_training.py
"""

import time
import warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

# ── Temporarily override episode counts so run_am_training does just 3 updates ─
import config as _cfg

_orig_get_ep = _cfg.get_episodes_per_node
_cfg.get_episodes_per_node = lambda n: 3 * _cfg.PPO_N_EPISODES_PER_UPDATE  # 3 updates worth
# Restore after test

from config import SEED, MAX_DURATION, DEVICE
from problem_data import load_matrices
from am_training import run_am_training

NUM_NODES = 10

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading matrices for N={NUM_NODES} ...")
    time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr, \
        ltr_stack, trucks_stack, avail_prob_arr, reward_global_p95 = load_matrices(NUM_NODES)

    print(f"time_matrix : type={type(time_matrix).__name__}  dtype={time_matrix.dtype}")
    assert isinstance(time_matrix, np.ndarray), "time_matrix must be ndarray after Task 1!"

    print(f"\nRunning 3 PPO updates × {_cfg.PPO_N_EPISODES_PER_UPDATE} ep/update ...\n")
    t0 = time.perf_counter()

    agent, critic, ep_rewards, ep_losses, log = run_am_training(
        time_matrix, rate_stack, loads_stack, distance_arr, diesel_arr,
        NUM_NODES,
        ltr_stack=ltr_stack,
        trucks_stack=trucks_stack,
        avail_prob_arr=avail_prob_arr,
        reward_global_p95=reward_global_p95,
    )

    elapsed   = time.perf_counter() - t0
    n_updates = len(log)
    per_update = elapsed / max(n_updates, 1)
    print(f"\nCompleted {n_updates} updates in {elapsed:.1f}s  "
          f"({per_update:.1f}s/update)\n")

    # Verify no NaN/Inf
    rewards_arr = np.array(ep_rewards, dtype=np.float32)
    nan_count   = int(np.isnan(rewards_arr).sum())
    inf_count   = int(np.isinf(rewards_arr).sum())
    print(f"Episode rewards: mean={rewards_arr.mean():.2f}  "
          f"std={rewards_arr.std():.2f}  NaN={nan_count}  Inf={inf_count}")

    losses_arr = np.array(ep_losses, dtype=np.float32)
    loss_nan   = int(np.isnan(losses_arr).sum())
    print(f"Episode losses : mean={losses_arr.mean():.4f}  NaN={loss_nan}")

    for entry in log:
        for k, v in entry.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                print(f"  WARNING: {k}={v} in update {entry['update']}")

    if nan_count > 0 or inf_count > 0 or loss_nan > 0:
        print("\nSMOKE TEST FAILED — NaN/Inf detected!")
        raise SystemExit(1)

    print("\nSmoke test PASSED.")
    print(f"  → {per_update:.1f}s per update  "
          f"({_cfg.PPO_N_EPISODES_PER_UPDATE} episodes/update, N={NUM_NODES})")

    _cfg.get_episodes_per_node = _orig_get_ep


if __name__ == "__main__":
    main()
