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

// Compute Y[K,N] = X[K,M] @ W[N,M].T (i.e. X @ W^T, where W is (out, in) row-major).
// Uses simdgroup_matrix with W loaded as transposed.
//   K = 8 streams, M = EMBD = 16, N = output dim
// Caller declares simdgroup_matrix tiles; we iterate inner dim.
inline void matmul_8xMxN(threadgroup const bfloat *X, device const bfloat *W,
                          threadgroup bfloat *Y, int N) {
    // Two-tile output along N dim: handle 8 cols at a time.
    for (int n_tile = 0; n_tile < N; n_tile += 8) {
        simdgroup_matrix<bfloat, 8, 8> acc(0);
        // Inner reduction over EMBD = 16 = 2 tiles of 8.
        for (int k_tile = 0; k_tile < EMBD; k_tile += 8) {
            simdgroup_matrix<bfloat, 8, 8> a, b;
            simdgroup_load(a, X + k_tile, EMBD);                      // (K=8, 8): rows 0..7 cols k_tile..k_tile+7
            // W is (N, EMBD) row-major; we want W[n_tile..n_tile+8, k_tile..k_tile+8] as a (K=8, N=8) matrix
            // for matmul, but we use transposed load to read it as if columns of W were rows.
            simdgroup_load(b, W + n_tile * EMBD + k_tile, EMBD, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        simdgroup_store(acc, Y + n_tile, N);
    }
}
"""

# Body: each TG handles 8 streams.
SOURCE = r"""
    threadgroup bfloat  tg_x[K * EMBD];
    threadgroup bfloat  tg_xr[K * EMBD];
    threadgroup bfloat  tg_q[K * EMBD];
    threadgroup bfloat  tg_kv_temp[K * EMBD * 2];     // k, v projections
    threadgroup bfloat  tg_attn_out[K * EMBD];
    threadgroup bfloat  tg_h[K * MLP_H];
    threadgroup float tg_logits[K * 32];            // 32 = VOCAB padded for the LM matmul output
    threadgroup int   tg_tok[K];
    threadgroup int   tg_pos[K];
    threadgroup uint  tg_rng[K];
    threadgroup bfloat  tg_kc[K * BLOCK * EMBD];
    threadgroup bfloat  tg_vc[K * BLOCK * EMBD];

    uint tg = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint stream = lane / (EMBD / 4);          // lane // 4 → stream id (0..7)
    uint elem4  = lane % (EMBD / 4);          // lane mod 4 → which 4-elem chunk in EMBD
    uint N_STEPS = n_steps[0];

    // Initialize per-stream state.
    if (lane < uint(K)) {
        tg_tok[lane] = BOS;
        tg_pos[lane] = 0;
        tg_rng[lane] = seeds[tg * K + lane];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = 0; step < N_STEPS; step++) {
        int tok_l = tg_tok[stream];
        int pos_l = tg_pos[stream];

        // Embed: each lane writes 4 elements of one stream.
        {
            uint base = elem4 * 4;
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            for (int e = 0; e < 4; e++) {
                xrow[base + e] = W[OFF_WTE + tok_l * EMBD + base + e]
                               + W[OFF_WPE + pos_l * EMBD + base + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RMSnorm — one lane per (stream, 4-elem chunk). Reduce within the 4
        // lanes that own a stream via simd_shuffle.
        // Each lane: sum its 4 elements squared, then sum across the 4 lanes.
        {
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            uint base = elem4 * 4;
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            // Sum across the 4 lanes that own this stream (lanes stream*4 .. stream*4+3).
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = bfloat(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Save residual and second rmsnorm.
        {
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            threadgroup bfloat *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrrow[base + e] = xrow[base + e];
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = bfloat(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QKV matmul via simdgroup_matrix.
        matmul_8xMxN(tg_x, W + OFF_WQ, tg_q,                      EMBD);
        matmul_8xMxN(tg_x, W + OFF_WK, tg_kv_temp,                EMBD);
        matmul_8xMxN(tg_x, W + OFF_WV, tg_kv_temp + K * EMBD,     EMBD);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Write k,v into per-stream KV cache at this stream's pos.
        {
            uint base = elem4 * 4;
            threadgroup bfloat *kc_s = &tg_kc[stream * BLOCK * EMBD + pos_l * EMBD];
            threadgroup bfloat *vc_s = &tg_vc[stream * BLOCK * EMBD + pos_l * EMBD];
            threadgroup bfloat *k_proj = &tg_kv_temp[stream * EMBD];
            threadgroup bfloat *v_proj = &tg_kv_temp[K * EMBD + stream * EMBD];
            for (int e = 0; e < 4; e++) {
                kc_s[base + e] = k_proj[base + e];
                vc_s[base + e] = v_proj[base + e];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int t_n = pos_l + 1;

        // Attention: per stream, per head. One head per lane within a stream
        // (4 lanes per stream × 1 head each = 4 heads, perfect).
        {
            threadgroup bfloat *q_s = &tg_q[stream * EMBD];
            threadgroup bfloat *kc_s = &tg_kc[stream * BLOCK * EMBD];
            threadgroup bfloat *vc_s = &tg_vc[stream * BLOCK * EMBD];
            threadgroup bfloat *out_s = &tg_attn_out[stream * EMBD];
            uint hi = elem4;  // 0..3 — head index
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
            out_s[hi*HD+0] = bfloat(o0);
            out_s[hi*HD+1] = bfloat(o1);
            out_s[hi*HD+2] = bfloat(o2);
            out_s[hi*HD+3] = bfloat(o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // WO + residual.
        matmul_8xMxN(tg_attn_out, W + OFF_WO, tg_x, EMBD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            threadgroup bfloat *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrow[base + e] += xrrow[base + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Save residual + RMSnorm.
        {
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            threadgroup bfloat *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrrow[base + e] = xrow[base + e];
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = bfloat(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc1: (K=8, EMBD=16) x (EMBD=16, MLP_H=64) -> (K=8, 64). 8 simdgroup tiles.
        matmul_8xMxN(tg_x, W + OFF_W1, tg_h, MLP_H);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ReLU per stream — each lane handles 8 of the 64 hidden elems.
        {
            threadgroup bfloat *hrow = &tg_h[stream * MLP_H];
            uint base = elem4 * 16;        // 4 lanes × 16 elems per lane = 64
            for (int e = 0; e < 16; e++) hrow[base + e] = max(hrow[base + e], bfloat(0));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MLP fc2: (K=8, MLP_H=64) x (MLP_H=64, EMBD=16) -> (K=8, EMBD=16).
        // matmul_8xMxN expects inner dim = EMBD; we need a different inline path.
        // Here we inline: 2 output tiles, 8 inner tiles (MLP_H=64 = 8*8).
        {
            for (int n_tile = 0; n_tile < EMBD; n_tile += 8) {
                simdgroup_matrix<bfloat, 8, 8> acc(0);
                for (int k_tile = 0; k_tile < MLP_H; k_tile += 8) {
                    simdgroup_matrix<bfloat, 8, 8> a, b;
                    simdgroup_load(a, &tg_h[k_tile], MLP_H);
                    simdgroup_load(b, W + OFF_W2 + n_tile * MLP_H + k_tile, MLP_H, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc, a, b, acc);
                }
                simdgroup_store(acc, &tg_x[n_tile], EMBD);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            threadgroup bfloat *xrrow = &tg_xr[stream * EMBD];
            uint base = elem4 * 4;
            for (int e = 0; e < 4; e++) xrow[base + e] += xrrow[base + e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // RMSnorm before LM head.
        {
            threadgroup bfloat *xrow = &tg_x[stream * EMBD];
            uint base = elem4 * 4;
            float sq = 0;
            for (int e = 0; e < 4; e++) sq += float(xrow[base + e]) * float(xrow[base + e]);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
            float scale = 1.0f / sqrt(sq * INV_EMBD + EPS);
            for (int e = 0; e < 4; e++) xrow[base + e] = bfloat(float(xrow[base + e]) * scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // LM head: (K=8, EMBD=16) x (VOCAB_PAD=32, EMBD=16) -> (K=8, 32).
        // We have 32 logit slots per stream; only first 27 are valid.
        // For correctness we need the LM weights padded to 32 rows (zeros for the extras).
        // We'll handle this on the host side by padding W_lm.
        {
            // 4 output tiles along N (32 = 4*8)
            for (int n_tile = 0; n_tile < 32; n_tile += 8) {
                simdgroup_matrix<bfloat, 8, 8> acc(0);
                for (int k_tile = 0; k_tile < EMBD; k_tile += 8) {
                    simdgroup_matrix<bfloat, 8, 8> a, b;
                    simdgroup_load(a, &tg_x[k_tile], EMBD);
                    simdgroup_load(b, W_lm_pad + n_tile * EMBD + k_tile, EMBD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(acc, a, b, acc);
                }
                // tg_logits is fp32; cast tile and store.
                threadgroup bfloat tile_h[8 * 8];
                simdgroup_store(acc, tile_h, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (lane < 32) {
                    uint s = lane / 4;
                    uint c = lane % 4;
                    tg_logits[s * 32 + n_tile + c * 2 + 0] = float(tile_h[s * 8 + c * 2 + 0]) / TEMP;
                    tg_logits[s * 32 + n_tile + c * 2 + 1] = float(tile_h[s * 8 + c * 2 + 1]) / TEMP;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        // Sample: lane = stream (use only lanes 0..7).
        if (lane < uint(K)) {
            uint s = lane;
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

    if (lane < uint(K)) seeds_out[tg * K + lane] = tg_rng[lane];
"""

KERNEL = mx.fast.metal_kernel(
    name="microgpt_streams_sg_bf16",
    input_names=["W", "W_lm_pad", "seeds", "n_steps"],
    output_names=["tokens", "seeds_out"],
    header=HEADER,
    source=SOURCE,
    ensure_row_contiguous=True,
)


def load_weights():
    """Returns (flat_fp32, lm_pad_fp32) numpy arrays. Caller converts to
    mx.bfloat16 via mx.array(..., dtype=mx.bfloat16)."""
    raw = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "assets/weights_only.npy"), allow_pickle=True).item()
    order = ["wte", "wpe",
             "layer0.attn_wq", "layer0.attn_wk", "layer0.attn_wv", "layer0.attn_wo",
             "layer0.mlp_fc1", "layer0.mlp_fc2", "lm_head"]
    flat = np.concatenate([raw[k].astype(np.float32).ravel() for k in order])
    lm_pad = np.zeros((32, EMBD), dtype=np.float32)
    lm_pad[:VOCAB] = raw["lm_head"].astype(np.float32)
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
    W = mx.array(W_flat).astype(mx.bfloat16)
    W_lm = mx.array(W_lm_pad).astype(mx.bfloat16)
    seeds = mx.array(np.arange(1, S + 1, dtype=np.uint32))
    n_steps = mx.array(np.array([N], dtype=np.uint32))

    def dispatch(seeds_in):
        outs = KERNEL(
            inputs=[W, W_lm, seeds_in, n_steps],
            grid=(N_TG * 32, 1, 1),
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
    label = f"mlx bf16+sg (gpu metal kernel S={S} N={N})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")


if __name__ == "__main__":
    main()
