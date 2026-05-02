"""MLX fp32 microGPT inference. Default device is CPU (matches single-thread baseline)."""

import argparse
import numpy as np
import mlx.core as mx
from model import (
    load_weights, make_sampler, benchmark,
    VOCAB_SIZE, BLOCK_SIZE, N_HEAD, N_EMBD, HEAD_DIM, BOS, TEMPERATURE,
)


def build(device):
    mx.set_default_device(device)
    W = load_weights()
    g = {k: mx.array(np.asarray(v)) for k, v in W.items()}
    inv_sqrt_hd = mx.array(np.float32(1.0 / np.sqrt(HEAD_DIM)))
    inv_temp = mx.array(np.float32(1.0 / TEMPERATURE))

    def rmsnorm(x):
        return x * mx.rsqrt((x * x).mean() + 1e-5)

    def forward(tok_arr, pos_arr, K, V, pos_int):
        x = g["wte"][tok_arr] + g["wpe"][pos_arr]
        x = rmsnorm(x)

        xr = x
        x = rmsnorm(x)
        q = g["layer0.attn_wq"] @ x
        k = g["layer0.attn_wk"] @ x
        v = g["layer0.attn_wv"] @ x

        # Update KV cache by writing this step's row.
        K[pos_int] = k
        V[pos_int] = v
        Kt = K[: pos_int + 1].reshape(pos_int + 1, N_HEAD, HEAD_DIM)
        Vt = V[: pos_int + 1].reshape(pos_int + 1, N_HEAD, HEAD_DIM)
        qh = q.reshape(N_HEAD, HEAD_DIM)
        logits = mx.einsum("hd,thd->ht", qh, Kt) * inv_sqrt_hd
        aw = mx.softmax(logits, axis=1)
        head_out = mx.einsum("ht,thd->hd", aw, Vt).reshape(N_EMBD)

        x = g["layer0.attn_wo"] @ head_out
        x = x + xr

        xr = x
        x = rmsnorm(x)
        h = g["layer0.mlp_fc1"] @ x
        h = mx.maximum(h, 0)
        x = g["layer0.mlp_fc2"] @ h
        x = x + xr

        logits_out = g["lm_head"] @ x
        logits_out = logits_out * inv_temp
        probs = mx.softmax(logits_out)
        return probs

    return g, forward


def make_step(device, seed=42):
    g, forward = build(device)
    sample = make_sampler(seed)
    K = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
    V = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
    state = {"pos": 0, "tok": BOS}

    def step():
        if state["pos"] >= BLOCK_SIZE:
            state["pos"] = 0
            state["tok"] = BOS
        tok_arr = mx.array(state["tok"])
        pos_arr = mx.array(state["pos"])
        probs = forward(tok_arr, pos_arr, K, V, state["pos"])
        mx.eval(probs, K, V)
        nxt = sample(np.asarray(probs).tolist())
        if nxt == BOS:
            state["pos"] = 0
            state["tok"] = BOS
        else:
            state["pos"] += 1
            state["tok"] = nxt
        return nxt

    return step


def sample_names(device, n=20, seed=42):
    import random
    g, forward = build(device)
    chars = sorted("abcdefghijklmnopqrstuvwxyz")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        K = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
        V = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
        tok = BOS
        s = []
        for pos in range(BLOCK_SIZE):
            probs = forward(mx.array(tok), mx.array(pos), K, V, pos)
            mx.eval(probs, K, V)
            tok = rng.choices(range(VOCAB_SIZE), weights=np.asarray(probs).tolist())[0]
            if tok == BOS:
                break
            s.append(chars[tok])
        out.append("".join(s))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    args = ap.parse_args()
    device = mx.gpu if args.gpu else mx.cpu
    label = f"mlx fp32 ({'gpu' if args.gpu else 'cpu'})"
    if args.names:
        for i, name in enumerate(sample_names(device, 20)):
            print(f"sample {i+1:2d}: {name}")
    else:
        benchmark(make_step(device), n=args.n, warmup=args.warmup, label=label)
