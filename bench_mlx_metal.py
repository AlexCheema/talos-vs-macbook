"""microGPT inference via a hand-written Metal kernel run through
mx.fast.metal_kernel. Ports tungsten-llama's microgpt_streams.metal:
each of S threadgroups (32 lanes) runs N_STEPS tokens of one
autoregressive stream entirely on-GPU. One Metal dispatch produces
S * N_STEPS tokens; no per-token host round-trip.

usage: bench_mlx_metal.py [--streams S] [--steps N] [--reps R]
"""
import os, sys, time, argparse
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--streams", type=int, default=1024)
ap.add_argument("--steps", type=int, default=256)
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--warmup", type=int, default=5)
args = ap.parse_args()

VOCAB = 27
BLOCK = 16
EMBD  = 16
HEAD  = 4
HD    = 4
MLP_H = 64

# Header: constants + the rmsnorm helper.
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

inline float rmsnorm_scale(float xv) {
    float sq = xv * xv;
    sq = simd_sum(sq);
    return 1.0f / metal::sqrt(sq * INV_EMBD + EPS);
}
"""

# Body — runs inside the kernel function body that mx.fast.metal_kernel
# generates. inputs are: W (float[4192]), seeds (uint[S]), n_steps (uint[1]).
# outputs: tokens (uint[S*N_STEPS]), K_pool (float[S*BLOCK*EMBD]),
#          V_pool (float[S*BLOCK*EMBD]), seeds_out (uint[S]).
SOURCE = r"""
    threadgroup float tg_x[EMBD];
    threadgroup float tg_xr[EMBD];
    threadgroup float tg_q[EMBD];
    threadgroup float tg_attn_out[EMBD];
    threadgroup float tg_h[MLP_H];
    threadgroup float tg_logits[VOCAB];
    threadgroup float tg_al[BLOCK * HEAD];
    threadgroup int   tg_tok;
    threadgroup int   tg_pos;
    threadgroup uint  tg_rng;

    uint stream = threadgroup_position_in_grid.x;
    uint lane   = thread_position_in_threadgroup.x;
    uint N_STEPS = n_steps[0];

    device float *kc = K_pool + stream * BLOCK * EMBD;
    device float *vc = V_pool + stream * BLOCK * EMBD;

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
            float v = tg_x[lane];
            tg_x[lane] = v * rmsnorm_scale(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) tg_xr[lane] = tg_x[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < uint(EMBD)) {
            float v = tg_x[lane];
            tg_x[lane] = v * rmsnorm_scale(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            float qv = 0.0f, kv = 0.0f, vv = 0.0f;
            int row_off = int(lane) * EMBD;
            for (int j = 0; j < EMBD; j++) {
                float xj = tg_x[j];
                qv += W[OFF_WQ + row_off + j] * xj;
                kv += W[OFF_WK + row_off + j] * xj;
                vv += W[OFF_WV + row_off + j] * xj;
            }
            tg_q[lane] = qv;
            kc[pos * EMBD + lane] = kv;
            vc[pos * EMBD + lane] = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int t_n = pos + 1;

        if (lane % HD == 0 && lane < uint(EMBD)) {
            int hi = int(lane) / HD;
            float maxl = -1e30f;
            for (int t = 0; t < t_n; t++) {
                int koff = t * EMBD + hi * HD;
                float dot = tg_q[hi*HD + 0] * kc[koff + 0]
                          + tg_q[hi*HD + 1] * kc[koff + 1]
                          + tg_q[hi*HD + 2] * kc[koff + 2]
                          + tg_q[hi*HD + 3] * kc[koff + 3];
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
                o += w * vc[t * EMBD + hi * HD + dim_in_head];
            }
            tg_attn_out[lane] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            float v = 0.0f;
            int row_off = int(lane) * EMBD;
            for (int j = 0; j < EMBD; j++) v += W[OFF_WO + row_off + j] * tg_attn_out[j];
            tg_x[lane] = v + tg_xr[lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) tg_xr[lane] = tg_x[lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < uint(EMBD)) {
            float v = tg_x[lane];
            tg_x[lane] = v * rmsnorm_scale(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc1: each of 32 lanes does 2 rows.
        {
            int row = int(lane) * 2;
            float a0 = 0.0f, a1 = 0.0f;
            for (int j = 0; j < EMBD; j++) {
                float xj = tg_x[j];
                a0 += W[OFF_W1 + (row + 0) * EMBD + j] * xj;
                a1 += W[OFF_W1 + (row + 1) * EMBD + j] * xj;
            }
            tg_h[row + 0] = metal::max(a0, 0.0f);
            tg_h[row + 1] = metal::max(a1, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            float v = 0.0f;
            int row_off = int(lane) * MLP_H;
            for (int j = 0; j < MLP_H; j++) v += W[OFF_W2 + row_off + j] * tg_h[j];
            tg_x[lane] = v + tg_xr[lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(EMBD)) {
            float v = tg_x[lane];
            tg_x[lane] = v * rmsnorm_scale(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lane < uint(VOCAB)) {
            float v = 0.0f;
            int row_off = int(lane) * EMBD;
            for (int j = 0; j < EMBD; j++) v += W[OFF_LM + row_off + j] * tg_x[j];
            tg_logits[lane] = v / TEMP;
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
            if (p >= BLOCK) { p = 0; tg_tok = BOS; }
            tg_pos = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) seeds_out[stream] = tg_rng;
"""

KERNEL = mx.fast.metal_kernel(
    name="microgpt_streams",
    input_names=["W", "seeds", "n_steps"],
    # K_pool/V_pool are outputs because MLX inputs are const-qualified and
    # the kernel writes the cache in place. The host throws them away.
    output_names=["tokens", "seeds_out", "K_pool", "V_pool"],
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
    return np.concatenate([raw[k].astype(np.float32).ravel() for k in order])


def main():
    S = args.streams
    N = args.steps

    mx.set_default_device(mx.gpu)
    W_flat = load_weights()
    assert W_flat.size == 4192, f"weights size {W_flat.size}"

    W = mx.array(W_flat)
    seeds = mx.array(np.arange(1, S + 1, dtype=np.uint32))
    n_steps = mx.array(np.array([N], dtype=np.uint32))

    kv_size = S * BLOCK * EMBD
    def dispatch(seeds_in):
        outs = KERNEL(
            inputs=[W, seeds_in, n_steps],
            grid=(S * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(S * N,), (S,), (kv_size,), (kv_size,)],
            output_dtypes=[mx.uint32, mx.uint32, mx.float32, mx.float32],
        )
        return outs[0], outs[1]

    # Warmup.
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
    label = f"mlx fp32 (gpu metal kernel S={S} N={N})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")


if __name__ == "__main__":
    main()
