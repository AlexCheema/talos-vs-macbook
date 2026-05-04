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
        # Inverse-CDF sample: matches Python's bisect(cumsum(probs), u),
        # including the fallback-to-last-token behavior when floating-point
        # roundoff leaves the final cumulative probability below u.
        cdf = mx.cumsum(probs)
        gt = cdf > u_arr
        nxt = mx.where(
            mx.any(gt),
            mx.argmax(gt.astype(mx.int32)),
            mx.array(VOCAB_SIZE - 1, dtype=mx.int32),
        )
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


def make_async_step(device, seed=42, rollout=1):
    """Async step: never syncs. Returns (step, flush, tokens_per_call).
    rollout > 1 unrolls R sequential token generations into one compiled
    dispatch — fewer Python iterations, fewer mx.array allocs per token,
    larger fused graph for MLX to optimize."""
    import random
    from functools import partial
    g, forward, forward_and_sample, _ = build(device)
    rng = random.Random(seed)
    state = {
        "tok": mx.array(BOS, dtype=mx.int32),
        "pos": mx.array(0, dtype=mx.int32),
        "K": mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32),
        "V": mx.zeros((BLOCK_SIZE, N_EMBD), dtype=mx.float32),
    }

    bos_arr = mx.array(BOS, dtype=mx.int32)
    zero_arr = mx.array(0, dtype=mx.int32)
    block_arr = mx.array(BLOCK_SIZE, dtype=mx.int32)

    def step_rollout(tok_in, pos_in, u_arr, K, V):
        """u_arr: (rollout,). Generates `rollout` tokens sequentially."""
        tok, pos = tok_in, pos_in
        for r in range(rollout):
            need_reset_pre = pos >= block_arr
            tok = mx.where(need_reset_pre, bos_arr, tok)
            pos = mx.where(need_reset_pre, zero_arr, pos)
            nxt, K, V = forward_and_sample(tok, pos, u_arr[r], K, V)
            nxt_i32 = nxt.astype(mx.int32)
            is_bos = nxt_i32 == bos_arr
            tok = mx.where(is_bos, bos_arr, nxt_i32)
            pos = mx.where(is_bos, zero_arr, pos + 1)
        return tok, pos, K, V

    step_rollout_compiled = mx.compile(step_rollout)

    def step():
        u = np.fromiter((rng.random() for _ in range(rollout)),
                        dtype=np.float32, count=rollout)
        u_arr = mx.array(u)
        nt, np_, K_new, V_new = step_rollout_compiled(
            state["tok"], state["pos"], u_arr, state["K"], state["V"]
        )
        state["tok"], state["pos"], state["K"], state["V"] = nt, np_, K_new, V_new

    def flush():
        mx.eval(state["tok"], state["pos"], state["K"], state["V"])

    return step, flush, rollout


def benchmark_async(step, flush, n, warmup, label, flush_every=64, tokens_per_call=1):
    """n is the requested *token* count; total tokens generated = n_calls * tokens_per_call.
    flush_every is in calls, not tokens — flushing too often serializes the pipeline."""
    import time
    n_calls = (n + tokens_per_call - 1) // tokens_per_call
    warmup_calls = (warmup + tokens_per_call - 1) // tokens_per_call
    for i in range(warmup_calls):
        step()
        if (i + 1) % flush_every == 0:
            flush()
    flush()
    t0 = time.perf_counter()
    for i in range(n_calls):
        step()
        if (i + 1) % flush_every == 0:
            flush()
    flush()
    t1 = time.perf_counter()
    total_tokens = n_calls * tokens_per_call
    rate = total_tokens / (t1 - t0)
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")
    return rate


def build_batched(device, batch_size):
    """Batched forward + step graph. Each batch item is an independent stream."""
    mx.set_default_device(device)
    W = load_weights()
    g = {k: mx.array(np.asarray(v)) for k, v in W.items()}
    inv_sqrt_hd = float(1.0 / np.sqrt(HEAD_DIM))
    inv_temp = mx.array(np.float32(1.0 / TEMPERATURE))
    positions = mx.arange(BLOCK_SIZE)
    bos_arr = mx.array(BOS, dtype=mx.int32)
    zero_arr = mx.array(0, dtype=mx.int32)
    block_arr = mx.array(BLOCK_SIZE, dtype=mx.int32)

    # Pre-transpose weights so we can write `x @ W_T` for batched matmul.
    Wq_T = g["layer0.attn_wq"].T
    Wk_T = g["layer0.attn_wk"].T
    Wv_T = g["layer0.attn_wv"].T
    Wo_T = g["layer0.attn_wo"].T
    Wfc1_T = g["layer0.mlp_fc1"].T
    Wfc2_T = g["layer0.mlp_fc2"].T
    Wlm_T = g["lm_head"].T

    def forward_b(tok_arr, pos_arr, K, V):
        # tok_arr, pos_arr: (B,) int32. K, V: (B, BLOCK_SIZE, N_EMBD).
        x = g["wte"][tok_arr] + g["wpe"][pos_arr]            # (B, N_EMBD)
        x = mx.fast.rms_norm(x, weight=None, eps=1e-5)

        xr = x
        x = mx.fast.rms_norm(x, weight=None, eps=1e-5)
        q = x @ Wq_T                                          # (B, N_EMBD)
        k = x @ Wk_T
        v = x @ Wv_T

        one_hot = (positions.reshape(1, BLOCK_SIZE) == pos_arr.reshape(batch_size, 1)) \
            .astype(mx.float32).reshape(batch_size, BLOCK_SIZE, 1)
        K_new = K * (1.0 - one_hot) + one_hot * k.reshape(batch_size, 1, N_EMBD)
        V_new = V * (1.0 - one_hot) + one_hot * v.reshape(batch_size, 1, N_EMBD)

        # SDPA shapes: (B, N_HEAD, T, HEAD_DIM). T_q=1, T_kv=BLOCK_SIZE.
        q_in = q.reshape(batch_size, 1, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        k_in = K_new.reshape(batch_size, BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        v_in = V_new.reshape(batch_size, BLOCK_SIZE, N_HEAD, HEAD_DIM).transpose(0, 2, 1, 3)
        attn_mask = (positions.reshape(1, 1, 1, BLOCK_SIZE) <= pos_arr.reshape(batch_size, 1, 1, 1))
        attn_out = mx.fast.scaled_dot_product_attention(q_in, k_in, v_in, scale=inv_sqrt_hd, mask=attn_mask)
        head_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, N_EMBD)

        x = head_out @ Wo_T
        x = x + xr

        xr = x
        x = mx.fast.rms_norm(x, weight=None, eps=1e-5)
        h = x @ Wfc1_T
        h = mx.maximum(h, 0)
        x = h @ Wfc2_T
        x = x + xr

        logits_out = (x @ Wlm_T) * inv_temp                   # (B, VOCAB_SIZE)
        probs = mx.softmax(logits_out, axis=-1)
        return probs, K_new, V_new

    def step_graph_b(tok_in, pos_in, u_arr, K, V):
        # All ops elementwise / per-batch-item.
        need_reset_pre = pos_in >= block_arr
        tok = mx.where(need_reset_pre, bos_arr, tok_in)
        pos = mx.where(need_reset_pre, zero_arr, pos_in)
        probs, K_new, V_new = forward_b(tok, pos, K, V)
        cdf = mx.cumsum(probs, axis=-1)                       # (B, VOCAB_SIZE)
        mask = cdf > u_arr.reshape(batch_size, 1)
        # Per-row fallback to VOCAB_SIZE - 1 on FP roundoff.
        any_above = mx.any(mask, axis=-1)
        nxt = mx.where(any_above,
                       mx.argmax(mask.astype(mx.int32), axis=-1),
                       mx.array(VOCAB_SIZE - 1, dtype=mx.int32))
        nxt_i32 = nxt.astype(mx.int32)
        is_bos = nxt_i32 == bos_arr
        next_tok = mx.where(is_bos, bos_arr, nxt_i32)
        next_pos = mx.where(is_bos, zero_arr, pos + 1)
        return next_tok, next_pos, K_new, V_new

    return mx.compile(forward_b), mx.compile(step_graph_b)


def make_batch_step(device, batch_size, seed=42, rollout=1):
    """Async-style batched step: never syncs. Returns (step, flush, tokens_per_call).
    If rollout > 1, each call does rollout sequential token gens per stream
    (B * rollout tokens per call) inside one compiled dispatch."""
    import random
    forward_b, step_graph_b = build_batched(device, batch_size)
    rng = random.Random(seed)
    state = {
        "tok": mx.full((batch_size,), BOS, dtype=mx.int32),
        "pos": mx.zeros((batch_size,), dtype=mx.int32),
        "K": mx.zeros((batch_size, BLOCK_SIZE, N_EMBD), dtype=mx.float32),
        "V": mx.zeros((batch_size, BLOCK_SIZE, N_EMBD), dtype=mx.float32),
    }

    if rollout == 1:
        def step():
            u = np.fromiter((rng.random() for _ in range(batch_size)),
                            dtype=np.float32, count=batch_size)
            u_arr = mx.array(u)
            nt, np_, K_new, V_new = step_graph_b(state["tok"], state["pos"], u_arr, state["K"], state["V"])
            state["tok"], state["pos"], state["K"], state["V"] = nt, np_, K_new, V_new
    else:
        def step_rollout_b(tok, pos, u_arr, K, V):
            # u_arr: (rollout, batch_size)
            for r in range(rollout):
                tok, pos, K, V = step_graph_b(tok, pos, u_arr[r], K, V)
            return tok, pos, K, V

        step_rollout_b_compiled = mx.compile(step_rollout_b)

        def step():
            u = np.fromiter((rng.random() for _ in range(rollout * batch_size)),
                            dtype=np.float32, count=rollout * batch_size).reshape(rollout, batch_size)
            u_arr = mx.array(u)
            nt, np_, K_new, V_new = step_rollout_b_compiled(
                state["tok"], state["pos"], u_arr, state["K"], state["V"]
            )
            state["tok"], state["pos"], state["K"], state["V"] = nt, np_, K_new, V_new

    def flush():
        mx.eval(state["tok"], state["pos"], state["K"], state["V"])

    return step, flush, batch_size * rollout


def benchmark_batch(step, flush, tokens_per_call, n_steps, warmup_steps, label, flush_every=64):
    """Reports total tokens/sec. n_steps is the *call* count; total tokens = n_steps * tokens_per_call.
    flush_every is in calls, not tokens — flushing too often serializes the pipeline."""
    import time
    for i in range(warmup_steps):
        step()
        if (i + 1) % flush_every == 0:
            flush()
    flush()
    t0 = time.perf_counter()
    for i in range(n_steps):
        step()
        if (i + 1) % flush_every == 0:
            flush()
    flush()
    t1 = time.perf_counter()
    rate = n_steps * tokens_per_call / (t1 - t0)
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
    ap.add_argument("--batch", type=int, default=1,
                    help="Batch size: process N independent streams in parallel (always async)")
    ap.add_argument("--rollout", type=int, default=1,
                    help="Unroll R sequential token gens into one compiled dispatch (async only)")
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--warmup", type=int, default=2_000)
    args = ap.parse_args()
    device = mx.gpu if args.gpu else mx.cpu
    dev_name = "gpu" if args.gpu else "cpu"
    if args.names:
        for i, name in enumerate(sample_names(device, 20)):
            print(f"sample {i+1:2d}: {name}")
    elif args.batch > 1:
        suffix = f" batch={args.batch}" if args.rollout == 1 else f" batch={args.batch} r={args.rollout}"
        label = f"mlx fp32 ({dev_name}{suffix})"
        step, flush, tpc = make_batch_step(device, args.batch, rollout=args.rollout)
        benchmark_batch(step, flush, tpc, n_steps=args.n, warmup_steps=args.warmup, label=label)
    elif args.async_mode:
        suffix = f" async" if args.rollout == 1 else f" async r={args.rollout}"
        label = f"mlx fp32 ({dev_name}{suffix})"
        step, flush, tpc = make_async_step(device, rollout=args.rollout)
        benchmark_async(step, flush, n=args.n, warmup=args.warmup, label=label, tokens_per_call=tpc)
    else:
        label = f"mlx fp32 ({dev_name})"
        benchmark(make_step(device), n=args.n, warmup=args.warmup, label=label)
