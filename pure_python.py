"""Karpathy's microGPT inference, loaded with TALOS weights, dependency-free Python.

This is the slow baseline. Same forward function as assets/microgpt.py, just
with the trained-vs-loaded distinction stripped out.
"""

import math
import argparse
import numpy as np
from model import (
    load_weights, make_sampler, benchmark,
    VOCAB_SIZE, BLOCK_SIZE, N_LAYER, N_HEAD, N_EMBD, HEAD_DIM, BOS, TEMPERATURE,
)

W = load_weights()
# Convert each weight to a list-of-lists so the inner loops are pure Python.
WLL = {k: v.tolist() for k, v in W.items()}


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values):
    tok = WLL["wte"][token_id]
    pos = WLL["wpe"][pos_id]
    x = [a + b for a, b in zip(tok, pos)]
    x = rmsnorm(x)
    for li in range(N_LAYER):
        xr = x
        x = rmsnorm(x)
        q = linear(x, WLL[f"layer{li}.attn_wq"])
        k = linear(x, WLL[f"layer{li}.attn_wk"])
        v = linear(x, WLL[f"layer{li}.attn_wv"])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            qh = q[hs:hs + HEAD_DIM]
            kh = [ki[hs:hs + HEAD_DIM] for ki in keys[li]]
            vh = [vi[hs:hs + HEAD_DIM] for vi in values[li]]
            scale = HEAD_DIM ** 0.5
            al = [sum(qh[j] * kh[t][j] for j in range(HEAD_DIM)) / scale for t in range(len(kh))]
            aw = softmax(al)
            ho = [sum(aw[t] * vh[t][j] for t in range(len(vh))) for j in range(HEAD_DIM)]
            x_attn.extend(ho)
        x = linear(x_attn, WLL[f"layer{li}.attn_wo"])
        x = [a + b for a, b in zip(x, xr)]
        xr = x
        x = rmsnorm(x)
        x = linear(x, WLL[f"layer{li}.mlp_fc1"])
        x = [v if v > 0 else 0.0 for v in x]
        x = linear(x, WLL[f"layer{li}.mlp_fc2"])
        x = [a + b for a, b in zip(x, xr)]
    return linear(x, WLL["lm_head"])


def make_step(seed=42):
    sample = make_sampler(seed)
    state = {"keys": [[]], "values": [[]], "pos": 0, "tok": BOS}

    def step():
        if state["pos"] >= BLOCK_SIZE:
            state["keys"] = [[]]
            state["values"] = [[]]
            state["pos"] = 0
            state["tok"] = BOS
        logits = gpt(state["tok"], state["pos"], state["keys"], state["values"])
        scaled = [l / TEMPERATURE for l in logits]
        probs = softmax(scaled)
        nxt = sample(probs)
        if nxt == BOS:
            state["keys"] = [[]]
            state["values"] = [[]]
            state["pos"] = 0
            state["tok"] = BOS
        else:
            state["pos"] += 1
            state["tok"] = nxt
        return nxt

    return step


def sample_names(n=20, seed=42):
    """Generate names exactly the way assets/microgpt.py does, for correctness check."""
    import random
    chars = sorted("abcdefghijklmnopqrstuvwxyz")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        keys, values = [[]], [[]]
        tok = BOS
        s = []
        for pos in range(BLOCK_SIZE):
            logits = gpt(tok, pos, keys, values)
            probs = softmax([l / TEMPERATURE for l in logits])
            tok = rng.choices(range(VOCAB_SIZE), weights=probs)[0]
            if tok == BOS:
                break
            s.append(chars[tok])
        out.append("".join(s))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", action="store_true", help="print sample names instead of benchmarking")
    ap.add_argument("--n", type=int, default=2_000)
    ap.add_argument("--warmup", type=int, default=200)
    args = ap.parse_args()
    if args.names:
        for i, name in enumerate(sample_names(20)):
            print(f"sample {i+1:2d}: {name}")
    else:
        benchmark(make_step(), n=args.n, warmup=args.warmup, label="pure-python")
