// microGPT inference, single-thread, batch=1, NEON-optimized fp32.
// Weights laid out per WEIGHT_ORDER in model.py.
//
// build: clang -O3 -march=native -ffast-math bench_c.c -o bench_c
// run:   ./bench_c [N_TOKENS] [WARMUP]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <arm_neon.h>

#define VOCAB 27
#define BLOCK 16
#define EMBD 16
#define HEAD 4
#define HD 4
#define MLP_H 64
#define BOS 26
#define TEMP 0.5f

// Weight offsets in the flat fp32 buffer (in #floats).
#define OFF_WTE 0
#define OFF_WPE (OFF_WTE + VOCAB * EMBD)        // 432
#define OFF_WQ  (OFF_WPE + BLOCK * EMBD)        // 688
#define OFF_WK  (OFF_WQ  + EMBD * EMBD)         // 944
#define OFF_WV  (OFF_WK  + EMBD * EMBD)         // 1200
#define OFF_WO  (OFF_WV  + EMBD * EMBD)         // 1456
#define OFF_W1  (OFF_WO  + EMBD * EMBD)         // 1712
#define OFF_W2  (OFF_W1  + MLP_H * EMBD)        // 2736
#define OFF_LM  (OFF_W2  + EMBD * MLP_H)        // 3760
#define TOTAL   (OFF_LM  + VOCAB * EMBD)        // 4192

static float W[TOTAL];

// xorshift32 RNG — deterministic per seed, sub-nanosecond per call.
static uint32_t rng_state = 42;
static inline uint32_t xrand(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}
static inline float urand(void) { return (xrand() >> 8) * (1.0f / (1u << 24)); }

// y = W @ x where W is (R, EMBD=16) row-major. Hand-unrolled with NEON.
// Inputs are 16-aligned in our setup (stack arrays, 16-element vectors).
static inline void matvec_16in(const float *Wm, const float *x, float *y, int R) {
    float32x4_t x0 = vld1q_f32(x +  0);
    float32x4_t x1 = vld1q_f32(x +  4);
    float32x4_t x2 = vld1q_f32(x +  8);
    float32x4_t x3 = vld1q_f32(x + 12);
    for (int r = 0; r < R; r++) {
        const float *wr = Wm + r * EMBD;
        float32x4_t a = vmulq_f32(vld1q_f32(wr +  0), x0);
        a = vfmaq_f32(a, vld1q_f32(wr +  4), x1);
        a = vfmaq_f32(a, vld1q_f32(wr +  8), x2);
        a = vfmaq_f32(a, vld1q_f32(wr + 12), x3);
        y[r] = vaddvq_f32(a);
    }
}

// y = W @ x where W is (EMBD=16, MLP_H=64) row-major.
static inline void matvec_mlp_out(const float *Wm, const float *x, float *y) {
    for (int r = 0; r < EMBD; r++) {
        const float *wr = Wm + r * MLP_H;
        float32x4_t a0 = vmulq_f32(vld1q_f32(wr +  0),  vld1q_f32(x +  0));
        float32x4_t a1 = vmulq_f32(vld1q_f32(wr +  4),  vld1q_f32(x +  4));
        float32x4_t a2 = vmulq_f32(vld1q_f32(wr +  8),  vld1q_f32(x +  8));
        float32x4_t a3 = vmulq_f32(vld1q_f32(wr + 12),  vld1q_f32(x + 12));
        a0 = vfmaq_f32(a0, vld1q_f32(wr + 16), vld1q_f32(x + 16));
        a1 = vfmaq_f32(a1, vld1q_f32(wr + 20), vld1q_f32(x + 20));
        a2 = vfmaq_f32(a2, vld1q_f32(wr + 24), vld1q_f32(x + 24));
        a3 = vfmaq_f32(a3, vld1q_f32(wr + 28), vld1q_f32(x + 28));
        a0 = vfmaq_f32(a0, vld1q_f32(wr + 32), vld1q_f32(x + 32));
        a1 = vfmaq_f32(a1, vld1q_f32(wr + 36), vld1q_f32(x + 36));
        a2 = vfmaq_f32(a2, vld1q_f32(wr + 40), vld1q_f32(x + 40));
        a3 = vfmaq_f32(a3, vld1q_f32(wr + 44), vld1q_f32(x + 44));
        a0 = vfmaq_f32(a0, vld1q_f32(wr + 48), vld1q_f32(x + 48));
        a1 = vfmaq_f32(a1, vld1q_f32(wr + 52), vld1q_f32(x + 52));
        a2 = vfmaq_f32(a2, vld1q_f32(wr + 56), vld1q_f32(x + 56));
        a3 = vfmaq_f32(a3, vld1q_f32(wr + 60), vld1q_f32(x + 60));
        y[r] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
    }
}

static inline void rmsnorm(float *x) {
    float32x4_t a = vmulq_f32(vld1q_f32(x +  0), vld1q_f32(x +  0));
    a = vfmaq_f32(a, vld1q_f32(x +  4), vld1q_f32(x +  4));
    a = vfmaq_f32(a, vld1q_f32(x +  8), vld1q_f32(x +  8));
    a = vfmaq_f32(a, vld1q_f32(x + 12), vld1q_f32(x + 12));
    float ms = vaddvq_f32(a) / EMBD;
    float scale = 1.0f / sqrtf(ms + 1e-5f);
    float32x4_t s = vdupq_n_f32(scale);
    vst1q_f32(x +  0, vmulq_f32(vld1q_f32(x +  0), s));
    vst1q_f32(x +  4, vmulq_f32(vld1q_f32(x +  4), s));
    vst1q_f32(x +  8, vmulq_f32(vld1q_f32(x +  8), s));
    vst1q_f32(x + 12, vmulq_f32(vld1q_f32(x + 12), s));
}

// Sample a token id from a probability vector via cumulative scan.
static inline int sample_probs(const float *p) {
    float r = urand();
    float c = 0.0f;
    for (int i = 0; i < VOCAB - 1; i++) {
        c += p[i];
        if (r < c) return i;
    }
    return VOCAB - 1;
}

// Forward one token. K, V are (BLOCK, EMBD) cache buffers.
static inline int step(int tok, int pos, float *K, float *V) {
    float x[EMBD] __attribute__((aligned(16)));
    float xr[EMBD] __attribute__((aligned(16)));
    float q[EMBD] __attribute__((aligned(16)));
    float k[EMBD] __attribute__((aligned(16)));
    float v[EMBD] __attribute__((aligned(16)));
    float h[MLP_H] __attribute__((aligned(16)));
    float head_out[EMBD] __attribute__((aligned(16)));
    float logits[VOCAB];

    const float *wte = W + OFF_WTE + tok * EMBD;
    const float *wpe = W + OFF_WPE + pos * EMBD;
    for (int i = 0; i < EMBD; i += 4) {
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(wte + i), vld1q_f32(wpe + i)));
    }
    rmsnorm(x);

    memcpy(xr, x, sizeof(x));
    rmsnorm(x);

    matvec_16in(W + OFF_WQ, x, q, EMBD);
    matvec_16in(W + OFF_WK, x, k, EMBD);
    matvec_16in(W + OFF_WV, x, v, EMBD);

    memcpy(K + pos * EMBD, k, sizeof(k));
    memcpy(V + pos * EMBD, v, sizeof(v));

    const float scale = 1.0f / 2.0f; // sqrt(HD=4) = 2
    int t_n = pos + 1;
    for (int hi = 0; hi < HEAD; hi++) {
        float *qh = q + hi * HD;
        float al[BLOCK];
        float maxl = -1e30f;
        for (int t = 0; t < t_n; t++) {
            const float *kh = K + t * EMBD + hi * HD;
            float dot = qh[0]*kh[0] + qh[1]*kh[1] + qh[2]*kh[2] + qh[3]*kh[3];
            al[t] = dot * scale;
            if (al[t] > maxl) maxl = al[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < t_n; t++) {
            al[t] = expf(al[t] - maxl);
            sum += al[t];
        }
        float inv = 1.0f / sum;
        float o0=0, o1=0, o2=0, o3=0;
        for (int t = 0; t < t_n; t++) {
            float w = al[t] * inv;
            const float *vh = V + t * EMBD + hi * HD;
            o0 += w * vh[0]; o1 += w * vh[1]; o2 += w * vh[2]; o3 += w * vh[3];
        }
        head_out[hi*HD+0] = o0; head_out[hi*HD+1] = o1;
        head_out[hi*HD+2] = o2; head_out[hi*HD+3] = o3;
    }

    matvec_16in(W + OFF_WO, head_out, x, EMBD);
    for (int i = 0; i < EMBD; i += 4) {
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(xr + i)));
    }

    memcpy(xr, x, sizeof(x));
    rmsnorm(x);

    matvec_16in(W + OFF_W1, x, h, MLP_H);
    for (int i = 0; i < MLP_H; i += 4) {
        float32x4_t z = vld1q_f32(h + i);
        vst1q_f32(h + i, vmaxq_f32(z, vdupq_n_f32(0.0f)));
    }
    matvec_mlp_out(W + OFF_W2, h, x);
    for (int i = 0; i < EMBD; i += 4) {
        vst1q_f32(x + i, vaddq_f32(vld1q_f32(x + i), vld1q_f32(xr + i)));
    }

    matvec_16in(W + OFF_LM, x, logits, VOCAB);
    float maxl = -1e30f;
    for (int i = 0; i < VOCAB; i++) {
        logits[i] /= TEMP;
        if (logits[i] > maxl) maxl = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < VOCAB; i++) {
        logits[i] = expf(logits[i] - maxl);
        sum += logits[i];
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < VOCAB; i++) logits[i] *= inv;

    return sample_probs(logits);
}

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(W, sizeof(float), TOTAL, f) != TOTAL) {
        fprintf(stderr, "short read\n"); exit(1);
    }
    fclose(f);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    long N = (argc > 1) ? atol(argv[1]) : 5000000;
    long WUP = (argc > 2) ? atol(argv[2]) : 100000;
    int names_mode = (argc > 1 && strcmp(argv[1], "--names") == 0);

    load_weights("assets/weights_fp32.bin");

    if (names_mode) {
        const char chars[] = "abcdefghijklmnopqrstuvwxyz";
        for (int s = 0; s < 20; s++) {
            float K[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
            float V[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
            int tok = BOS;
            char buf[BLOCK + 1] = {0};
            int len = 0;
            for (int pos = 0; pos < BLOCK; pos++) {
                tok = step(tok, pos, K, V);
                if (tok == BOS) break;
                buf[len++] = chars[tok];
            }
            printf("sample %2d: %s\n", s + 1, buf);
        }
        return 0;
    }

    float K[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
    float V[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
    int tok = BOS, pos = 0;

    for (long i = 0; i < WUP; i++) {
        if (pos >= BLOCK) { tok = BOS; pos = 0; }
        int nxt = step(tok, pos, K, V);
        if (nxt == BOS) { tok = BOS; pos = 0; }
        else { tok = nxt; pos++; }
    }
    double t0 = now_sec();
    long emitted = 0;
    for (long i = 0; i < N; i++) {
        if (pos >= BLOCK) { tok = BOS; pos = 0; }
        int nxt = step(tok, pos, K, V);
        emitted++;
        if (nxt == BOS) { tok = BOS; pos = 0; }
        else { tok = nxt; pos++; }
    }
    double t1 = now_sec();
    double rate = emitted / (t1 - t0);
    printf("  c fp32+NEON              %14.0f tok/sec\n", rate);
    return 0;
}
