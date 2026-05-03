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


# ---------- batched path ----------

POSITIONS = np.arange(BLOCK_SIZE, dtype=np.int32)


def forward_batched(tok, pos, K, V):
    """tok, pos: (B,) int. K, V: (B, BLOCK_SIZE, N_EMBD)."""
    B = tok.shape[0]
    x = WTE[tok] + WPE[pos]                                    # (B, N_EMBD)
    # rmsnorm over last axis
    ms = (x * x).mean(axis=-1, keepdims=True)
    x = x / np.sqrt(ms + np.float32(1e-5))

    xr = x
    ms = (x * x).mean(axis=-1, keepdims=True)
    x = x / np.sqrt(ms + np.float32(1e-5))
    q = x @ WQ.T                                               # (B, N_EMBD)
    k = x @ WK.T
    v = x @ WV.T

    # Shape-stable KV update.
    one_hot = (POSITIONS[None, :] == pos[:, None]).astype(np.float32)[:, :, None]
    K = K * (1.0 - one_hot) + one_hot * k[:, None, :]
    V = V * (1.0 - one_hot) + one_hot * v[:, None, :]

    # Per-head attention with broadcast.
    qh = q.reshape(B, N_HEAD, HEAD_DIM)                        # (B, H, D)
    Kh = K.reshape(B, BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)  # (B, H, T, D)
    Vh = V.reshape(B, BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    # logits (B, H, T) = qh (B,H,D) . Kh (B,H,T,D)
    logits = np.einsum("bhd,bhtd->bht", qh, Kh) * INV_SQRT_HD
    mask = (POSITIONS[None, None, :] <= pos[:, None, None])    # (B, 1, T)
    logits = np.where(mask, logits, np.float32(-1e9))
    logits -= logits.max(axis=-1, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=-1, keepdims=True)
    head_out = np.einsum("bht,bhtd->bhd", logits, Vh).reshape(B, N_EMBD)

    x = head_out @ WO.T
    x = x + xr

    xr = x
    ms = (x * x).mean(axis=-1, keepdims=True)
    x = x / np.sqrt(ms + np.float32(1e-5))
    h = x @ W1.T
    np.maximum(h, 0, out=h)
    x = h @ W2.T
    x = x + xr

    out = (x @ LM.T) * INV_TEMP                                # (B, VOCAB)
    out -= out.max(axis=-1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=-1, keepdims=True)
    return out, K, V


def benchmark_batch(batch_size, n_steps, warmup_steps, seed=42):
    import time, random
    rng = random.Random(seed)
    tok = np.full((batch_size,), BOS, dtype=np.int32)
    pos = np.zeros((batch_size,), dtype=np.int32)
    K = np.zeros((batch_size, BLOCK_SIZE, N_EMBD), dtype=np.float32)
    V = np.zeros((batch_size, BLOCK_SIZE, N_EMBD), dtype=np.float32)

    def one_step(tok, pos, K, V):
        # Pre-sample reset.
        reset = pos >= BLOCK_SIZE
        tok = np.where(reset, BOS, tok)
        pos = np.where(reset, 0, pos)
        probs, K, V = forward_batched(tok, pos, K, V)
        # Inverse-CDF sampling, vectorized.
        u = np.fromiter((rng.random() for _ in range(batch_size)),
                        dtype=np.float32, count=batch_size)
        cdf = np.cumsum(probs, axis=-1)
        nxt = (cdf > u[:, None]).argmax(axis=-1).astype(np.int32)
        is_bos = nxt == BOS
        next_tok = np.where(is_bos, BOS, nxt)
        next_pos = np.where(is_bos, 0, pos + 1)
        return next_tok, next_pos, K, V

    for _ in range(warmup_steps):
        tok, pos, K, V = one_step(tok, pos, K, V)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        tok, pos, K, V = one_step(tok, pos, K, V)
    t1 = time.perf_counter()
    rate = batch_size * n_steps / (t1 - t0)
    print(f"  numpy fp32 (batch={batch_size:<3d})    {rate:>14,.0f} tok/sec")
    return rate


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
    ap.add_argument("--batch", type=int, default=1, help="Batch size: process N independent streams in parallel")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=20_000)
    args = ap.parse_args()
    if args.names:
        for i, name in enumerate(sample_names(20)):
            print(f"sample {i+1:2d}: {name}")
    elif args.batch > 1:
        benchmark_batch(args.batch, n_steps=args.n, warmup_steps=args.warmup)
    else:
        benchmark(make_step(), n=args.n, warmup=args.warmup, label="numpy fp32")
