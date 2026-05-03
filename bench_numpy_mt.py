"""NumPy multi-threaded batched throughput.

Two parallelism options exposed:
  --blas-threads N : let BLAS use N internal threads on the matmul.
  --workers     T  : split B streams across T Python threads,
                     each calling numpy serially. numpy releases the
                     GIL during BLAS calls so this can scale even with
                     blas_threads=1 per worker.
"""

import os
import sys
import time
import math
import argparse
import threading

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--blas-threads", type=int, default=1)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--warmup", type=int, default=200)
args = ap.parse_args()

# Set BLAS thread count BEFORE numpy import.
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(args.blas_threads)

import numpy as np

VOCAB_SIZE = 27
BLOCK_SIZE = 16
N_HEAD = 4
N_EMBD = 16
HEAD_DIM = N_EMBD // N_HEAD
MLP_HIDDEN = 4 * N_EMBD
BOS = 26
TEMPERATURE = 0.5

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
W = np.load(os.path.join(ASSETS, "weights_only.npy"), allow_pickle=True).item()
WTE = W["wte"].astype(np.float32);  WPE = W["wpe"].astype(np.float32)
WQ  = W["layer0.attn_wq"].astype(np.float32); WK = W["layer0.attn_wk"].astype(np.float32)
WV  = W["layer0.attn_wv"].astype(np.float32); WO = W["layer0.attn_wo"].astype(np.float32)
W1  = W["layer0.mlp_fc1"].astype(np.float32); W2 = W["layer0.mlp_fc2"].astype(np.float32)
LM  = W["lm_head"].astype(np.float32)

INV_SQRT_HD = np.float32(1.0 / math.sqrt(HEAD_DIM))
INV_TEMP    = np.float32(1.0 / TEMPERATURE)
POSITIONS   = np.arange(BLOCK_SIZE, dtype=np.int32)


def forward_batched(tok, pos, K, V):
    B = tok.shape[0]
    x = WTE[tok] + WPE[pos]
    ms = (x * x).mean(axis=-1, keepdims=True)
    x = x / np.sqrt(ms + np.float32(1e-5))

    xr = x
    ms = (x * x).mean(axis=-1, keepdims=True)
    x = x / np.sqrt(ms + np.float32(1e-5))
    q = x @ WQ.T; k = x @ WK.T; v = x @ WV.T

    one_hot = (POSITIONS[None, :] == pos[:, None]).astype(np.float32)[:, :, None]
    K = K * (1.0 - one_hot) + one_hot * k[:, None, :]
    V = V * (1.0 - one_hot) + one_hot * v[:, None, :]

    qh = q.reshape(B, N_HEAD, HEAD_DIM)
    Kh = K.reshape(B, BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    Vh = V.reshape(B, BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
    logits = np.einsum("bhd,bhtd->bht", qh, Kh) * INV_SQRT_HD
    mask = (POSITIONS[None, None, :] <= pos[:, None, None])
    logits = np.where(mask, logits, np.float32(-1e9))
    logits -= logits.max(axis=-1, keepdims=True)
    np.exp(logits, out=logits)
    logits /= logits.sum(axis=-1, keepdims=True)
    head_out = np.einsum("bht,bhtd->bhd", logits, Vh).reshape(B, N_EMBD)

    x = head_out @ WO.T + xr
    xr = x
    ms = (x * x).mean(axis=-1, keepdims=True)
    x = x / np.sqrt(ms + np.float32(1e-5))
    h = x @ W1.T; np.maximum(h, 0, out=h)
    x = h @ W2.T + xr

    out = (x @ LM.T) * INV_TEMP
    out -= out.max(axis=-1, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=-1, keepdims=True)
    return out, K, V


class Worker(threading.Thread):
    def __init__(self, tid, B_local, n, warmup, start_bar, end_bar):
        super().__init__()
        self.tid = tid
        self.B_local = B_local
        self.n = n; self.warmup = warmup
        self.start_bar = start_bar
        self.end_bar = end_bar
        import random
        self.rng = random.Random(42 + tid * B_local)

    def step(self, tok, pos, K, V):
        reset = pos >= BLOCK_SIZE
        tok = np.where(reset, BOS, tok)
        pos = np.where(reset, 0, pos)
        probs, K, V = forward_batched(tok, pos, K, V)
        u = np.fromiter((self.rng.random() for _ in range(self.B_local)),
                        dtype=np.float32, count=self.B_local)
        cdf = np.cumsum(probs, axis=-1)
        nxt = (cdf > u[:, None]).argmax(axis=-1).astype(np.int32)
        is_bos = nxt == BOS
        next_tok = np.where(is_bos, BOS, nxt)
        next_pos = np.where(is_bos, 0, pos + 1)
        return next_tok, next_pos, K, V

    def run(self):
        B = self.B_local
        tok = np.full((B,), BOS, dtype=np.int32)
        pos = np.zeros((B,), dtype=np.int32)
        K = np.zeros((B, BLOCK_SIZE, N_EMBD), dtype=np.float32)
        V = np.zeros((B, BLOCK_SIZE, N_EMBD), dtype=np.float32)

        for _ in range(self.warmup):
            tok, pos, K, V = self.step(tok, pos, K, V)

        self.start_bar.wait()  # all workers + main meet here, then main records t0
        for _ in range(self.n):
            tok, pos, K, V = self.step(tok, pos, K, V)
        self.end_bar.wait()    # all workers + main meet here, then main records t1


def main():
    if args.batch % args.workers:
        sys.exit("--batch must be multiple of --workers")
    B_local = args.batch // args.workers

    start_bar = threading.Barrier(args.workers + 1)
    end_bar = threading.Barrier(args.workers + 1)
    workers = [Worker(t, B_local, args.n, args.warmup, start_bar, end_bar)
               for t in range(args.workers)]
    for w in workers: w.start()
    start_bar.wait()
    t0 = time.perf_counter()
    end_bar.wait()
    t1 = time.perf_counter()
    for w in workers: w.join()

    rate = args.batch * args.n / (t1 - t0)
    label = f"numpy fp32 (batch={args.batch} t={args.workers} blas={args.blas_threads})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")


main()
