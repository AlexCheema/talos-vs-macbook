"""Shared utilities: weight loading, sampler, benchmark harness.

All implementations target Karpathy's microGPT as shipped in TALOS-V2:
  vocab=27, block=16, n_layer=1, n_head=4, n_embd=16, head_dim=4,
  RMSNorm (no learnable gain), ReLU MLP (4x), no biases, untied lm_head.
"""

import os
import time
import random
import numpy as np

VOCAB_SIZE = 27
BLOCK_SIZE = 16
N_LAYER = 1
N_HEAD = 4
N_EMBD = 16
HEAD_DIM = N_EMBD // N_HEAD
MLP_HIDDEN = 4 * N_EMBD
BOS = 26
TEMPERATURE = 0.5

# Order matches convert_weights.py and the C benchmarks.
WEIGHT_ORDER = (
    "wte", "wpe",
    "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo",
    "layer0.mlp_fc1", "layer0.mlp_fc2",
    "lm_head",
)

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


def load_weights():
    raw = np.load(os.path.join(ASSETS, "weights_only.npy"), allow_pickle=True).item()
    return {k: np.asarray(raw[k], dtype=np.float32) for k in WEIGHT_ORDER}


def make_sampler(seed=42):
    """Return sample(probs)->int that pulls from a freshly-seeded RNG.

    Uses Python's random.choices so NumPy and MLX produce identical streams
    given identical probability vectors.
    """
    rng = random.Random(seed)
    pop = list(range(VOCAB_SIZE))
    def sample(probs):
        return rng.choices(pop, weights=probs)[0]
    return sample


def benchmark(step_fn, n=200_000, warmup=20_000, label=""):
    """step_fn() emits one token and returns it. We don't care what it is here.

    Returns tok/sec on the timed window only.
    """
    for _ in range(warmup):
        step_fn()
    t0 = time.perf_counter()
    for _ in range(n):
        step_fn()
    t1 = time.perf_counter()
    rate = n / (t1 - t0)
    if label:
        print(f"  {label:24s}  {rate:>14,.0f} tok/sec")
    return rate
