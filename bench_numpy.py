"""NumPy fp32 microGPT inference, single-thread, batch=1, with KV cache."""

import os
# Pin BLAS threads BEFORE importing numpy.
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import argparse
import numpy as np
from model import (
    load_weights, make_sampler, benchmark,
    VOCAB_SIZE, BLOCK_SIZE, N_HEAD, N_EMBD, HEAD_DIM, BOS, TEMPERATURE,
)

W = load_weights()
WTE = W["wte"]; WPE = W["wpe"]
WQ = W["layer0.attn_wq"]; WK = W["layer0.attn_wk"]
WV = W["layer0.attn_wv"]; WO = W["layer0.attn_wo"]
W1 = W["layer0.mlp_fc1"]; W2 = W["layer0.mlp_fc2"]
LM = W["lm_head"]

INV_SQRT_HD = 1.0 / np.sqrt(HEAD_DIM).astype(np.float32)
INV_TEMP = np.float32(1.0 / TEMPERATURE)


def rmsnorm(x):
    return x * np.float32(1.0) / np.sqrt((x * x).mean() + np.float32(1e-5))


class State:
    __slots__ = ("K", "V", "pos", "tok")

    def __init__(self):
        self.K = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
        self.V = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
        self.pos = 0
        self.tok = BOS

    def reset(self):
        self.pos = 0
        self.tok = BOS


def forward(tok, pos, K, V):
    x = WTE[tok] + WPE[pos]
    x = rmsnorm(x)

    xr = x
    x = rmsnorm(x)
    q = WQ @ x
    k = WK @ x
    v = WV @ x
    K[pos] = k
    V[pos] = v

    Kt = K[: pos + 1].reshape(pos + 1, N_HEAD, HEAD_DIM)
    Vt = V[: pos + 1].reshape(pos + 1, N_HEAD, HEAD_DIM)
    qh = q.reshape(N_HEAD, HEAD_DIM)
    # logits[h, t] = qh[h] . Kt[t, h]
    logits = np.einsum("hd,thd->ht", qh, Kt) * INV_SQRT_HD
    logits -= logits.max(axis=1, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=1, keepdims=True)
    # head_out[h, d] = sum_t logits[h,t] * Vt[t,h,d]
    head_out = np.einsum("ht,thd->hd", logits, Vt).reshape(N_EMBD)

    x = WO @ head_out
    x += xr

    xr = x
    x = rmsnorm(x)
    h = W1 @ x
    np.maximum(h, 0, out=h)
    x = W2 @ h
    x += xr

    return LM @ x


def make_step(seed=42):
    sample = make_sampler(seed)
    s = State()

    def step():
        if s.pos >= BLOCK_SIZE:
            s.reset()
        logits = forward(s.tok, s.pos, s.K, s.V)
        logits *= INV_TEMP
        logits -= logits.max()
        np.exp(logits, out=logits)
        logits /= logits.sum()
        nxt = sample(logits.tolist())
        if nxt == BOS:
            s.reset()
        else:
            s.pos += 1
            s.tok = nxt
        return nxt

    return step


def sample_names(n=20, seed=42):
    import random
    chars = sorted("abcdefghijklmnopqrstuvwxyz")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        K = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
        V = np.zeros((BLOCK_SIZE, N_EMBD), dtype=np.float32)
        tok = BOS
        s = []
        for pos in range(BLOCK_SIZE):
            logits = forward(tok, pos, K, V).copy()
            logits *= INV_TEMP
            logits -= logits.max()
            np.exp(logits, out=logits)
            logits /= logits.sum()
            tok = rng.choices(range(VOCAB_SIZE), weights=logits.tolist())[0]
            if tok == BOS:
                break
            s.append(chars[tok])
        out.append("".join(s))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    args = ap.parse_args()
    if args.names:
        for i, name in enumerate(sample_names(20)):
            print(f"sample {i+1:2d}: {name}")
    else:
        benchmark(make_step(), n=args.n, warmup=args.warmup, label="numpy fp32")
