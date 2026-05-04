"""microGPT inference via a hand-written Metal kernel — fp16 storage,
fp32 accumulators. Same in-kernel autoregressive loop as
bench_mlx_metal.py; only difference is precision.

Weights are loaded as fp16 once. K/V cache is fp16 (halves DRAM
traffic for the cache). Threadgroup scratch (x/xr/q/etc.) is fp16
since each value is small. Dot-product accumulators stay fp32 for
precision; the final logits softmax also accumulates in fp32. Sample
RNG is uint32 as before.
"""
import os, sys, time, argparse
import numpy as np
import mlx.core as mx

VOCAB = 27
BLOCK = 16
EMBD  = 16
HEAD  = 4
HD    = 4
MLP_H = 64

HEADER = r"""
constant int VOCAB = 27;
constant int BLOCK = 16;
constant int EMBD  = 16;
constant int HEAD  = 4;
constant int HD    = 4;
constant int MLP_H = 64;

constant int OFF_WTE = 0;
constant int OFF_WPE = 432;
constant int OFF_WQ  = 688;
constant int OFF_WK  = 944;
constant int OFF_WV  = 1200;
constant int OFF_WO  = 1456;
constant int OFF_W1  = 1712;
constant int OFF_W2  = 2736;
constant int OFF_LM  = 3760;

constant float ATTN_SCALE = 0.5f;
constant float INV_EMBD   = 0.0625f;
constant float EPS        = 1e-5f;
constant float TEMP       = 0.5f;
constant int   BOS        = 26;

// rmsnorm — operate in fp32 for the reduction, store back as half.
inline float rmsnorm_scale_f(float xv) {
    float sq = xv * xv;
    sq = simd_sum(sq);
    return 1.0f / metal::sqrt(sq * INV_EMBD + EPS);
}
"""

SOURCE = r"""
    threadgroup half  tg_x[EMBD];
    threadgroup half  tg_xr[EMBD];
    threadgroup half  tg_q[EMBD];
    threadgroup half  tg_attn_out[EMBD];
    threadgroup half  tg_h[MLP_H];
    threadgroup float tg_logits[VOCAB];          // fp32 for stable softmax
    threadgroup float tg_al[BLOCK * HEAD];       // fp32 for attention softmax
    threadgroup int   tg_tok;
    threadgroup int   tg_pos;
    threadgroup uint  tg_rng;

    uint stream = threadgroup_position_in_grid.x;
    uint lane   = thread_position_in_threadgroup.x;
    uint N_STEPS = n_steps[0];

    // KV cache lives in threadgroup memory (per-TG = per-stream). Avoids
    // DRAM r/w for K/V -- they stay in L1-equivalent on chip.
    // Size: 2 * BLOCK * EMBD * 2 bytes = 1 KB per TG.
    threadgroup half kc[BLOCK * EMBD];
    threadgroup half vc[BLOCK * EMBD];

    if (lane == 0) {
        tg_tok = BOS;
        tg_pos = 0;
        tg_rng = seeds[stream];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < N_STEPS; step++) {
        int tok = tg_tok;
        int pos = tg_pos;

        if (lane < uint(EMBD)) {
            tg_x[lane] = W[OFF_WTE + tok * EMBD + lane] + W[OFF_WPE + pos * EMBD + lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            float v = float(tg_x[lane]);
            tg_x[lane] = half(v * rmsnorm_scale_f(v));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) tg_xr[lane] = tg_x[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < uint(EMBD)) {
            float v = float(tg_x[lane]);
            tg_x[lane] = half(v * rmsnorm_scale_f(v));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QKV projections via half4-vectorized dot: 4 mul-adds per Metal instr.
        if (lane < uint(EMBD)) {
            int row_off = int(lane) * EMBD;
            threadgroup const half4 *xv4 = (threadgroup const half4 *)&tg_x[0];
            device    const half4 *Wq4 = (device const half4 *)(W + OFF_WQ + row_off);
            device    const half4 *Wk4 = (device const half4 *)(W + OFF_WK + row_off);
            device    const half4 *Wv4 = (device const half4 *)(W + OFF_WV + row_off);
            half4 q_acc = half4(0), k_acc = half4(0), v_acc = half4(0);
            for (int j = 0; j < EMBD / 4; j++) {
                half4 x4 = xv4[j];
                q_acc += Wq4[j] * x4;
                k_acc += Wk4[j] * x4;
                v_acc += Wv4[j] * x4;
            }
            tg_q[lane] = q_acc.x + q_acc.y + q_acc.z + q_acc.w;
            kc[pos * EMBD + lane] = k_acc.x + k_acc.y + k_acc.z + k_acc.w;
            vc[pos * EMBD + lane] = v_acc.x + v_acc.y + v_acc.z + v_acc.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int t_n = pos + 1;

        // Attention softmax per head.
        if (lane % HD == 0 && lane < uint(EMBD)) {
            int hi = int(lane) / HD;
            float maxl = -1e30f;
            for (int t = 0; t < t_n; t++) {
                int koff = t * EMBD + hi * HD;
                float dot = float(tg_q[hi*HD + 0]) * float(kc[koff + 0])
                          + float(tg_q[hi*HD + 1]) * float(kc[koff + 1])
                          + float(tg_q[hi*HD + 2]) * float(kc[koff + 2])
                          + float(tg_q[hi*HD + 3]) * float(kc[koff + 3]);
                float val = dot * ATTN_SCALE;
                tg_al[hi * BLOCK + t] = val;
                if (val > maxl) maxl = val;
            }
            float s = 0.0f;
            for (int t = 0; t < t_n; t++) {
                float e = metal::exp(tg_al[hi * BLOCK + t] - maxl);
                tg_al[hi * BLOCK + t] = e;
                s += e;
            }
            float inv = 1.0f / s;
            for (int t = 0; t < t_n; t++) tg_al[hi * BLOCK + t] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            int hi = int(lane) / HD;
            int dim_in_head = int(lane) % HD;
            float o = 0.0f;
            for (int t = 0; t < t_n; t++) {
                float w = tg_al[hi * BLOCK + t];
                o += w * float(vc[t * EMBD + hi * HD + dim_in_head]);
            }
            tg_attn_out[lane] = half(o);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            int row_off = int(lane) * EMBD;
            threadgroup const half4 *av4 = (threadgroup const half4 *)&tg_attn_out[0];
            device    const half4 *Wo4 = (device const half4 *)(W + OFF_WO + row_off);
            half4 acc = half4(0);
            for (int j = 0; j < EMBD / 4; j++) acc += Wo4[j] * av4[j];
            tg_x[lane] = (acc.x + acc.y + acc.z + acc.w) + tg_xr[lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) tg_xr[lane] = tg_x[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < uint(EMBD)) {
            float v = float(tg_x[lane]);
            tg_x[lane] = half(v * rmsnorm_scale_f(v));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc1: each of 32 lanes does 2 rows, half4-vectorized.
        {
            int row = int(lane) * 2;
            threadgroup const half4 *xv4 = (threadgroup const half4 *)&tg_x[0];
            device    const half4 *W1a = (device const half4 *)(W + OFF_W1 + (row + 0) * EMBD);
            device    const half4 *W1b = (device const half4 *)(W + OFF_W1 + (row + 1) * EMBD);
            half4 a0 = half4(0), a1 = half4(0);
            for (int j = 0; j < EMBD / 4; j++) {
                half4 x4 = xv4[j];
                a0 += W1a[j] * x4;
                a1 += W1b[j] * x4;
            }
            half s0 = a0.x + a0.y + a0.z + a0.w;
            half s1 = a1.x + a1.y + a1.z + a1.w;
            tg_h[row + 0] = metal::max(s0, half(0));
            tg_h[row + 1] = metal::max(s1, half(0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc2: each of 16 lanes does 1 row of MLP_H=64. 16 half4 chunks.
        if (lane < uint(EMBD)) {
            int row_off = int(lane) * MLP_H;
            threadgroup const half4 *hv4 = (threadgroup const half4 *)&tg_h[0];
            device    const half4 *W2v = (device const half4 *)(W + OFF_W2 + row_off);
            half4 acc = half4(0);
            for (int j = 0; j < MLP_H / 4; j++) acc += W2v[j] * hv4[j];
            tg_x[lane] = (acc.x + acc.y + acc.z + acc.w) + tg_xr[lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            float v = float(tg_x[lane]);
            tg_x[lane] = half(v * rmsnorm_scale_f(v));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(VOCAB)) {
            int row_off = int(lane) * EMBD;
            threadgroup const half4 *xv4 = (threadgroup const half4 *)&tg_x[0];
            device    const half4 *Lv4 = (device const half4 *)(W + OFF_LM + row_off);
            half4 acc = half4(0);
            for (int j = 0; j < EMBD / 4; j++) acc += Lv4[j] * xv4[j];
            tg_logits[lane] = float(acc.x + acc.y + acc.z + acc.w) / TEMP;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane == 0) {
            float maxl = tg_logits[0];
            for (int i = 1; i < VOCAB; i++) if (tg_logits[i] > maxl) maxl = tg_logits[i];
            float s = 0.0f;
            for (int i = 0; i < VOCAB; i++) {
                float e = metal::exp(tg_logits[i] - maxl);
                tg_logits[i] = e;
                s += e;
            }
            float inv = 1.0f / s;
            uint x = tg_rng;
            x ^= x << 13;  x ^= x >> 17;  x ^= x << 5;
            tg_rng = x;
            float r = float((x >> 8) & 0xFFFFFFu) * (1.0f / float(1u << 24));
            float c = 0.0f;
            int picked = VOCAB - 1;
            for (int i = 0; i < VOCAB - 1; i++) {
                c += tg_logits[i] * inv;
                if (r < c) { picked = i; break; }
            }
            tokens[stream * N_STEPS + step] = uint(picked);
            tg_tok = picked;
            int p = pos + 1;
            // Match Python/C BOS-reset: also reset pos when sampled token == BOS.
            if (picked == BOS || p >= BLOCK) { p = 0; tg_tok = BOS; }
            tg_pos = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) seeds_out[stream] = tg_rng;
"""

KERNEL = mx.fast.metal_kernel(
    name="microgpt_streams_tgkv",
    input_names=["W", "seeds", "n_steps"],
    output_names=["tokens", "seeds_out"],
    header=HEADER,
    source=SOURCE,
    ensure_row_contiguous=True,
)


def load_weights():
    raw = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "assets/weights_only.npy"), allow_pickle=True).item()
    order = ["wte", "wpe",
             "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo",
             "layer0.mlp_fc1", "layer0.mlp_fc2", "lm_head"]
    return np.concatenate([raw[k].astype(np.float32).ravel() for k in order]).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    S = args.streams
    N = args.steps

    mx.set_default_device(mx.gpu)
    W = mx.array(load_weights())
    assert W.size == 4192
    seeds = mx.array(np.arange(1, S + 1, dtype=np.uint32))
    n_steps = mx.array(np.array([N], dtype=np.uint32))

    def dispatch(seeds_in):
        outs = KERNEL(
            inputs=[W, seeds_in, n_steps],
            grid=(S * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(S * N,), (S,)],
            output_dtypes=[mx.uint32, mx.uint32],
        )
        return outs[0], outs[1]

    cur_seeds = seeds
    for _ in range(args.warmup):
        toks, cur_seeds = dispatch(cur_seeds)
        mx.eval(toks, cur_seeds)

    t0 = time.perf_counter()
    total_tokens = 0
    for _ in range(args.reps):
        toks, cur_seeds = dispatch(cur_seeds)
        mx.eval(toks, cur_seeds)
        total_tokens += S * N
    t1 = time.perf_counter()

    rate = total_tokens / (t1 - t0)
    label = f"mlx fp16+simd+tgkv (gpu metal kernel S={S} N={N})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")


if __name__ == "__main__":
    main()
