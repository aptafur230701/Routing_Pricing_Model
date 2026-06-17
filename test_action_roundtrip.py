"""
test_action_roundtrip.py
========================
Validates that the float32 cast path used in the CURRENT act_with_value
(action_t.view(1).float() → .cpu().numpy() → int()) round-trips correctly
against action_t.item() for 1000 samples from a real AttentionDecoder.

Also verifies the PROPOSED fix (separate .item() for action) is equivalent.

Usage: python3 test_action_roundtrip.py
"""

import numpy as np
import torch
from attention_decoder import AttentionDecoder

N       = 20
D_H     = 128
N_HEADS = 8
CLIP_C  = 10.0
DEVICE  = torch.device("cpu")
N_SAMPLES = 1000

decoder = AttentionDecoder(D_H, n_heads_glimpse=N_HEADS, clip_C=CLIP_C).to(DEVICE)
decoder.eval()

rng = np.random.default_rng(0)
torch.manual_seed(0)

n_fail_float32 = 0
n_fail_item    = 0

for i in range(N_SAMPLES):
    # Random context and embeddings
    context    = torch.randn(1, D_H, device=DEVICE)
    embeddings = torch.randn(1, N, D_H, device=DEVICE)

    # Random mask — at least 2 valid nodes (never all-zero)
    mask_np    = np.zeros(N, dtype=np.int8)
    valid_idx  = rng.choice(N, size=rng.integers(2, N), replace=False)
    mask_np[valid_idx] = 1
    mask_t     = torch.from_numpy(mask_np).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        action_t, log_prob, entropy = decoder.act(context, embeddings, mask_t)

    # Ground truth: int64 → Python int via .item()
    action_true = action_t.item()

    # Path A: float32 cast (current implementation)
    sync = torch.stack([
        action_t.view(1).float(),
        log_prob.view(1),
        torch.tensor([0.0]),  # placeholder for critic_val
    ]).cpu().numpy()
    action_float32 = int(sync[0])
    if action_float32 != action_true:
        n_fail_float32 += 1
        print(f"  FAIL float32 path @ sample {i}: action_true={action_true}  reconstructed={action_float32}")

    # Path B: separate .item() (proposed fix)
    action_item = int(action_t.item())
    if action_item != action_true:
        n_fail_item += 1
        print(f"  FAIL .item() path @ sample {i}: should never happen")

print(f"\nSamples tested : {N_SAMPLES}")
print(f"float32 path   : {n_fail_float32} failures  ({'PASS' if n_fail_float32 == 0 else 'FAIL'})")
print(f".item() path   : {n_fail_item} failures  ({'PASS' if n_fail_item == 0 else 'FAIL'})")

assert n_fail_float32 == 0, f"float32 round-trip failed {n_fail_float32}/{N_SAMPLES} times"
assert n_fail_item    == 0, "item() path failed — impossible"

print("\nAll 1000 round-trip assertions PASSED.")
