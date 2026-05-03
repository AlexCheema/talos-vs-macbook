"""microGPT inference via Metal kernel using simdgroup_matrix tensor ops.

Restructured from 1-stream-per-TG to 8-streams-per-TG so each matmul
becomes (8, EMBD) x (EMBD, EMBD) = (8, EMBD) — a true matrix-matrix
multiply that fits Apple's 8x8 simdgroup_matrix tile primitive.

Layout:
  - 1 threadgroup = 8 streams = 1 simdgroup (32 threads)
  - Lane mapping for elementwise ops: lane = stream*4 + elem//4,
    each lane handles 4 EMBD elements of one stream
  - All matmuls (QKV, WO, W1, W2, LM) via simdgroup_multiply_accumulate
  - Attention, rmsnorm, sample loop scalar over the 8 streams in a TG
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
TOK_PER_TG = 8

HEADER = r"""
#include <metal_simdgroup_matrix>
using namespace metal;

constant int VOCAB = 27;
constant int BLOCK = 16;
constant int EMBD  = 16;
constant int HEAD  = 4;
constant int HD    = 4;
constant int MLP_H = 64;
constant int K     = 8;       // streams per TG (matches simdgroup_matrix size)

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

// Two-simdgroup-cooperative matmul. Caller passes sg_idx (0 or 1).
// Each simdgroup handles every other output tile along N, halving
// per-simdgroup work. M_inner is the inner reduction size.
inline void matmul_8xMxN_sg2(threadgroup const half *X, device const half *W,
                             threadgroup half *Y, int N, int M_inner,
                             uint sg_idx) {
    // Output tiles along N are 8-wide; we have N/8 tiles total.
    // Simdgroup 0 handles tiles 0, 2, 4, ...; simdgroup 1 handles 1, 3, 5, ...
    for (int n_tile = int(sg_idx) * 8; n_tile < N; n_tile += 16) {
        simdgroup_matrix<half, 8, 8> acc(0);
        for (int k_tile = 0; k_tile < M_inner; k_tile += 8) {
            simdgroup_matrix<half, 8, 8> a, b;
            simdgroup_load(a, X + k_tile, M_inner);
            simdgroup_load(b, W + n_tile * M_inner + k_tile, M_inner, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        simdgroup_store(acc, Y + n_tile, N);
    }
}
"""

# Body: each TG handles 8 streams.
SOURCE = r"""
    threadgroup half  tg_x[K * EMBD];
    threadgroup half  tg_xr[K * EMBD];
    threadgroup half  tg_q[K * EMBD];
    threadgroup half  tg_kv_temp[K * EMBD * 2];     // k, v projections
    threadgroup half  tg_attn_out[K * EMBD];
    threadgroup half  tg_h[K * MLP_H];
    threadgroup float tg_logits[K * 32];            // 32 = VOCAB padded for the LM matmul output
    threadgroup int   tg_tok[K];
    threadgroup int   tg_pos[K];
    threadgroup uint  tg_rng[K];
    threadgroup half  tg_kc[K * BLOCK * EMBD];
    threadgroup half  tg_vc[K * BLOCK * EMBD];
    threadgroup half  tg_logits_h[K * 32];          // staging for LM-head fp16 tile

    uint tg = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint sg_idx = lane / 32;          // 0 or 1 (2 simdgroups per TG)
    uint sg_lane = lane % 32;         // 0..31 within simdgroup
    // Elementwise/attention work uses only simdgroup 0's lanes (sg_lane).
    uint stream = sg_lane / (EMBD / 4);      // 0..7
    uint elem4  = sg_lane % (EMBD / 4);      // 0..3
    uint N_STEPS = n_steps[0];

    if (sg_idx == 0 && sg_lane < uint(K)) {
        tg_tok[sg_lane] = BOS;
        tg_pos[sg_lane] = 0;
        tg_rng[sg_lane] = seeds[tg * K + sg_lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < N_STEPS; step++) {
        int tok_l = tg_tok[stream];
        int pos_l = tg_pos[stream];

        if (sg_idx == 0) {
            // Embed.
            uint base = elem4 * 4;
            threadgroup half *xrow = &tg_x[stream * EMBD];
            for (int e = 0; e < 4; e++) {
                xrow[base + e] = W[OFF_WTE + tok_l * EMBD + base + e]
                               + W[OFF_WPE + pos_l * EMBD + base + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_idx == 0) {
            // RMSnorm 1.
            threadgroup half *xrow = &tg_x[stream * EMBD];
            uint base = elem4 * 4;
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = half(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_idx == 0) {
            // Save residual + RMSnorm 2.
            threadgroup half *xrow = &tg_x[stream * EMBD];
            threadgroup half *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrrow[base + e] = xrow[base + e];
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = half(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QKV matmul: 2 simdgroups split the 2 output tiles each.
        matmul_8xMxN_sg2(tg_x, W + OFF_WQ, tg_q,                  EMBD, EMBD, sg_idx);
        matmul_8xMxN_sg2(tg_x, W + OFF_WK, tg_kv_temp,            EMBD, EMBD, sg_idx);
        matmul_8xMxN_sg2(tg_x, W + OFF_WV, tg_kv_temp + K * EMBD, EMBD, EMBD, sg_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_idx == 0) {
            // KV cache update.
            uint base = elem4 * 4;
            threadgroup half *kc_s = &tg_kc[stream * BLOCK * EMBD + pos_l * EMBD];
            threadgroup half *vc_s = &tg_vc[stream * BLOCK * EMBD + pos_l * EMBD];
            threadgroup half *k_proj = &tg_kv_temp[stream * EMBD];
            threadgroup half *v_proj = &tg_kv_temp[K * EMBD + stream * EMBD];
            for (int e = 0; e < 4; e++) {
                kc_s[base + e] = k_proj[base + e];
                vc_s[base + e] = v_proj[base + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int t_n = pos_l + 1;

        if (sg_idx == 0) {
            // Attention: 8 streams * 4 heads = 32 work units, one per lane in sg 0.
            threadgroup half *q_s = &tg_q[stream * EMBD];
            threadgroup half *kc_s = &tg_kc[stream * BLOCK * EMBD];
            threadgroup half *vc_s = &tg_vc[stream * BLOCK * EMBD];
            threadgroup half *out_s = &tg_attn_out[stream * EMBD];
            uint hi = elem4;
            float al[BLOCK];
            float maxl = -1e30f;
            for (int t = 0; t < t_n; t++) {
                int koff = t * EMBD + hi * HD;
                float dot = float(q_s[hi*HD+0]) * float(kc_s[koff+0])
                          + float(q_s[hi*HD+1]) * float(kc_s[koff+1])
                          + float(q_s[hi*HD+2]) * float(kc_s[koff+2])
                          + float(q_s[hi*HD+3]) * float(kc_s[koff+3]);
                float val = dot * ATTN_SCALE;
                al[t] = val;
                if (val > maxl) maxl = val;
            }
            float s = 0.0f;
            for (int t = 0; t < t_n; t++) { al[t] = exp(al[t] - maxl); s += al[t]; }
            float inv = 1.0f / s;
            float o0=0, o1=0, o2=0, o3=0;
            for (int t = 0; t < t_n; t++) {
                float w = al[t] * inv;
                int voff = t * EMBD + hi * HD;
                o0 += w * float(vc_s[voff+0]);
                o1 += w * float(vc_s[voff+1]);
                o2 += w * float(vc_s[voff+2]);
                o3 += w * float(vc_s[voff+3]);
            }
            out_s[hi*HD+0] = half(o0);
            out_s[hi*HD+1] = half(o1);
            out_s[hi*HD+2] = half(o2);
            out_s[hi*HD+3] = half(o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // WO matmul (split across 2 simdgroups).
        matmul_8xMxN_sg2(tg_attn_out, W + OFF_WO, tg_x, EMBD, EMBD, sg_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_idx == 0) {
            threadgroup half *xrow = &tg_x[stream * EMBD];
            threadgroup half *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrow[base + e] += xrrow[base + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_idx == 0) {
            threadgroup half *xrow = &tg_x[stream * EMBD];
            threadgroup half *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrrow[base + e] = xrow[base + e];
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = half(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc1 (K=8, EMBD=16) x (EMBD=16, MLP_H=64).
        matmul_8xMxN_sg2(tg_x, W + OFF_W1, tg_h, MLP_H, EMBD, sg_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_idx == 0) {
            threadgroup half *hrow = &tg_h[stream * MLP_H];
            uint base = elem4 * 16;
            for (int e = 0; e < 16; e++) hrow[base + e] = max(hrow[base + e], half(0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc2: inner=MLP_H=64, output=EMBD=16. 2 output tiles, 8 inner tiles.
        matmul_8xMxN_sg2(tg_h, W + OFF_W2, tg_x, EMBD, MLP_H, sg_idx);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_idx == 0) {
            threadgroup half *xrow = &tg_x[stream * EMBD];
            threadgroup half *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrow[base + e] += xrrow[base + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RMSnorm before LM head.
        {
            threadgroup half *xrow = &tg_x[stream * EMBD];
            uint base = elem4 * 4;
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = half(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // LM head: (K=8, EMBD=16) x (VOCAB_PAD=32, EMBD=16) -> (K=8, 32).
        // LM head: both simdgroups split the 4 output tiles (32 cols / 8).
        for (int n_tile = int(sg_idx) * 8; n_tile < 32; n_tile += 16) {
            simdgroup_matrix<half, 8, 8> acc(0);
            for (int k_tile = 0; k_tile < EMBD; k_tile += 8) {
                simdgroup_matrix<half, 8, 8> a, b;
                simdgroup_load(a, &tg_x[k_tile], EMBD);
                simdgroup_load(b, W_lm_pad + n_tile * EMBD + k_tile, EMBD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, tg_logits_h + n_tile, 32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_idx == 0) {
            // Cast & scale into fp32 tg_logits (8 streams * 32 cols / 32 lanes = 8 cells/lane).
            uint s = sg_lane / 4;          // 0..7
            uint c4 = sg_lane % 4;          // 0..3 — handles cols c4*8 .. c4*8+7
            for (int e = 0; e < 8; e++) {
                tg_logits[s * 32 + c4 * 8 + e] = float(tg_logits_h[s * 32 + c4 * 8 + e]) / TEMP;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Sample: only sg 0's lanes 0..7.
        if (sg_idx == 0 && sg_lane < uint(K)) {
            uint s = sg_lane;
            threadgroup float *lg = &tg_logits[s * 32];
            // softmax over first VOCAB=27 entries.
            float maxl = lg[0];
            for (int i = 1; i < VOCAB; i++) if (lg[i] > maxl) maxl = lg[i];
            float ssum = 0;
            for (int i = 0; i < VOCAB; i++) { lg[i] = exp(lg[i] - maxl); ssum += lg[i]; }
            float inv = 1.0f / ssum;
            uint x = tg_rng[s];
            x ^= x << 13;  x ^= x >> 17;  x ^= x << 5;
            tg_rng[s] = x;
            float r = float((x >> 8) & 0xFFFFFFu) * (1.0f / float(1u << 24));
            float c = 0;
            int picked = VOCAB - 1;
            for (int i = 0; i < VOCAB - 1; i++) {
                c += lg[i] * inv;
                if (r < c) { picked = i; break; }
            }
            uint stream_global = tg * K + s;
            tokens[stream_global * N_STEPS + step] = uint(picked);
            tg_tok[s] = picked;
            int p = tg_pos[s] + 1;
            if (p >= BLOCK) { p = 0; tg_tok[s] = BOS; }
            tg_pos[s] = p;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_idx == 0 && sg_lane < uint(K)) seeds_out[tg * K + sg_lane] = tg_rng[sg_lane];
"""

KERNEL = mx.fast.metal_kernel(
    name="microgpt_streams_sg2",
    input_names=["W", "W_lm_pad", "seeds", "n_steps"],
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
    flat = np.concatenate([raw[k].astype(np.float32).ravel() for k in order]).astype(np.float16)
    # Pad LM head to 32 rows (last 5 are zeros).
    lm_pad = np.zeros((32, EMBD), dtype=np.float16)
    lm_pad[:VOCAB] = raw["lm_head"].astype(np.float16)
    return flat, lm_pad.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=8192,
                    help="Total streams; will be rounded up to multiple of TOK_PER_TG=8")
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    S = ((args.streams + TOK_PER_TG - 1) // TOK_PER_TG) * TOK_PER_TG
    N_TG = S // TOK_PER_TG
    N = args.steps

    mx.set_default_device(mx.gpu)
    W_flat, W_lm_pad = load_weights()
    W = mx.array(W_flat)
    W_lm = mx.array(W_lm_pad)
    seeds = mx.array(np.arange(1, S + 1, dtype=np.uint32))
    n_steps = mx.array(np.array([N], dtype=np.uint32))

    def dispatch(seeds_in):
        outs = KERNEL(
            inputs=[W, W_lm, seeds_in, n_steps],
            grid=(N_TG * 64, 1, 1),
            threadgroup=(64, 1, 1),
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
    label = f"mlx fp16+sg2 (gpu metal kernel S={S} N={N})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")


if __name__ == "__main__":
    main()
