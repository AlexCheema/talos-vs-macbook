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
    inv_sqrt_hd = float(1.0 / np.sqrt(HEAD_DIM))
    inv_temp = mx.array(np.float32(1.0 / TEMPERATURE))

    positions = mx.arange(BLOCK_SIZE)

    def forward(tok_arr, pos_arr, K, V):
        x = g["wte"][tok_arr] + g["wpe"][pos_arr]
        x = mx.fast.rms_norm(x, weight=None, eps=1e-5)

        xr = x
        x = mx.fast.rms_norm(x, weight=None, eps=1e-5)
        q = g["layer0.attn_wq"] @ x
        k = g["layer0.attn_wk"] @ x
        v = g["layer0.attn_wv"] @ x

        # Shape-stable KV update: scatter via one-hot mask so compile sees one shape.
        one_hot = (positions == pos_arr).astype(mx.float32).reshape(BLOCK_SIZE, 1)
        K_new = K * (1.0 - one_hot) + one_hot * k.reshape(1, N_EMBD)
        V_new = V * (1.0 - one_hot) + one_hot * v.reshape(1, N_EMBD)

        # Reshape to (B=1, N_heads, T, D) for fused SDPA.
        q_in = q.reshape(1, N_HEAD, 1, HEAD_DIM)
        k_in = K_new.reshape(BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(1, 0, 2).reshape(1, N_HEAD, BLOCK_SIZE, HEAD_DIM)
        v_in = V_new.reshape(BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(1, 0, 2).reshape(1, N_HEAD, BLOCK_SIZE, HEAD_DIM)
        attn_mask = (positions <= pos_arr).reshape(1, 1, 1, BLOCK_SIZE)
        attn_out = mx.fast.scaled_dot_product_attention(
            q_in, k_in, v_in, scale=inv_sqrt_hd, mask=attn_mask
        )
        head_out = attn_out.reshape(N_EMBD)

        x = g["layer0.attn_wo"] @ head_out
        x = x + xr

        xr = x
        x = mx.fast.rms_norm(x, weight=None, eps=1e-5)
        h = g["layer0.mlp_fc1"] @ x
        h = mx.maximum(h, 0)
        x = g["layer0.mlp_fc2"] @ h
        x = x + xr

        logits_out = g["lm_head"] @ x
        logits_out = logits_out * inv_temp
        probs = mx.softmax(logits_out)
        return probs, K_new, V_new

    def forward_and_sample(tok_arr, pos_arr, u_arr, K, V):
        probs, K_new, V_new = forward(tok_arr, pos_arr, K, V)
        # Inverse-CDF sample: matches Python's bisect(cumsum(probs), u).
        cdf = mx.cumsum(probs)
        nxt = mx.argmax((cdf > u_arr).astype(mx.int32))
        return nxt, K_new, V_new

    bos_arr = mx.array(BOS, dtype=mx.int32)
    zero_arr = mx.array(0, dtype=mx.int32)
    block_arr = mx.array(BLOCK_SIZE, dtype=mx.int32)

    def step_graph(tok_in, pos_in, u_arr, K, V):
        # Pre-sample reset: matches old "if state['pos'] >= BLOCK_SIZE".
        need_reset_pre = pos_in >= block_arr
        tok = mx.where(need_reset_pre, bos_arr, tok_in)
        pos = mx.where(need_reset_pre, zero_arr, pos_in)
        nxt, K_new, V_new = forward_and_sample(tok, pos, u_arr, K, V)
        nxt_i32 = nxt.astype(mx.int32)
        is_bos = nxt_i32 == bos_arr
        next_tok = mx.where(is_bos, bos_arr, nxt_i32)
        next_pos = mx.where(is_bos, zero_arr, pos + 1)
        return next_tok, next_pos, K_new, V_new

    forward_compiled = mx.compile(forward)
    forward_and_sample_compiled = mx.compile(forward_and_sample)
    step_graph_compiled = mx.compile(step_graph)
    return g, forward_compiled, forward_and_sample_compiled, step_graph_compiled


def make_step(device, seed=42):
    import random
    g, forward, forward_and_sample, _ = build(device)
    rng = random.Random(seed)
    K = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
    V = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
    state = {"pos": 0, "tok": BOS}

    cache = {"K": K, "V": V}

    def step():
        if state["pos"] >= BLOCK_SIZE:
            state["pos"] = 0
            state["tok"] = BOS
        tok_arr = mx.array(state["tok"])
        pos_arr = mx.array(state["pos"])
        u_arr = mx.array(np.float32(rng.random()))
        nxt_arr, K_new, V_new = forward_and_sample(tok_arr, pos_arr, u_arr, cache["K"], cache["V"])
        mx.eval(nxt_arr, K_new, V_new)
        cache["K"], cache["V"] = K_new, V_new
        nxt = nxt_arr.item()
        if nxt == BOS:
            state["pos"] = 0
            state["tok"] = BOS
        else:
            state["pos"] += 1
            state["tok"] = nxt
        return nxt

    return step


def make_async_step(device, seed=42):
    """Async step: never syncs. Returns (step, flush). Caller must flush periodically
    and once at end of timed window for memory safety and correct timing."""
    import random
    g, forward, forward_and_sample, step_graph = build(device)
    rng = random.Random(seed)
    state = {
        "tok": mx.array(BOS, dtype=mx.int32),
        "pos": mx.array(0, dtype=mx.int32),
        "K": mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32),
        "V": mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32),
    }

    def step():
        u_arr = mx.array(np.float32(rng.random()))
        nt, np_, K_new, V_new = step_graph(state["tok"], state["pos"], u_arr, state["K"], state["V"])
        state["tok"], state["pos"], state["K"], state["V"] = nt, np_, K_new, V_new

    def flush():
        mx.eval(state["tok"], state["pos"], state["K"], state["V"])

    return step, flush


def benchmark_async(step, flush, n, warmup, label, flush_every=64):
    import time
    for i in range(warmup):
        step()
        if (i + 1) % flush_every == 0:
            flush()
    flush()
    t0 = time.perf_counter()
    for i in range(n):
        step()
        if (i + 1) % flush_every == 0:
            flush()
    flush()
    t1 = time.perf_counter()
    rate = n / (t1 - t0)
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")
    return rate


def sample_names(device, n=20, seed=42):
    import random
    g, forward, _, _ = build(device)
    chars = sorted("abcdefghijklmnopqrstuvwxyz")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        K = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
        V = mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32)
        tok = BOS
        s = []
        for pos in range(BLOCK_SIZE):
            probs, K, V = forward(mx.array(tok), mx.array(pos), K, V)
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
    ap.add_argument("--async", dest="async_mode", action="store_true",
                    help="Async pipelining: no per-step sync, flush every 64 steps + at end")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    args = ap.parse_args()
    device = mx.gpu if args.gpu else mx.cpu
    suffix = " async" if args.async_mode else ""
    label = f"mlx fp32 ({'gpu' if args.gpu else 'cpu'}{suffix})"
    if args.names:
        for i, name in enumerate(sample_names(device, 20)):
            print(f"sample {i+1:2d}: {name}")
    elif args.async_mode:
        step, flush = make_async_step(device)
        benchmark_async(step, flush, n=args.n, warmup=args.warmup, label=label)
    else:
        benchmark(make_step(device), n=args.n, warmup=args.warmup, label=label)
