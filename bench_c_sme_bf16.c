// microGPT batched inference using Apple BNNS with bf16 matmul.
// Weights pre-converted to bf16 at startup; inputs cast to bf16 each step.
// Output stays in fp32. BNNS dispatches matmul to whatever is fastest
// for the given dtype on the chip — SME bf16 path on M5.
//
// build: clang -O3 -march=native -ffast-math bench_c_sme_bf16.c \
//          -o bench_c_sme_bf16 -framework Accelerate
// run:   ./bench_c_sme_bf16 BATCH_SIZE [N_STEPS] [WARMUP]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/sysctl.h>
#include <arm_neon.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#define VOCAB 27
#define BLOCK 16
#define EMBD  16
#define HEAD  4
#define HD    4
#define MLP_H 64
#define BOS   26
#define TEMP  0.5f

#define OFF_WTE 0
#define OFF_WPE (OFF_WTE + VOCAB * EMBD)
#define OFF_WQ  (OFF_WPE + BLOCK * EMBD)
#define OFF_WK  (OFF_WQ  + EMBD * EMBD)
#define OFF_WV  (OFF_WK  + EMBD * EMBD)
#define OFF_WO  (OFF_WV  + EMBD * EMBD)
#define OFF_W1  (OFF_WO  + EMBD * EMBD)
#define OFF_W2  (OFF_W1  + MLP_H * EMBD)
#define OFF_LM  (OFF_W2  + EMBD * MLP_H)
#define TOTAL   (OFF_LM  + VOCAB * EMBD)

static float W[TOTAL];
// bf16 copies of every weight matrix that participates in a matmul.
static uint16_t WQ_bf[EMBD * EMBD];
static uint16_t WK_bf[EMBD * EMBD];
static uint16_t WV_bf[EMBD * EMBD];
static uint16_t WO_bf[EMBD * EMBD];
static uint16_t W1_bf[MLP_H * EMBD];
static uint16_t W2_bf[EMBD * MLP_H];
static uint16_t LM_bf[VOCAB * EMBD];

// Truncating fp32 -> bf16 (drop the lower 16 bits of the fp32 mantissa).
// Round-to-nearest-even would be more accurate but truncation is fast and
// good enough for a tiny char-LM.
static inline void f32_to_bf16(const float *src, uint16_t *dst, size_t n) {
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32x4_t v = vreinterpretq_u32_f32(vld1q_f32(src + i));
        uint16x4_t hi = vshrn_n_u32(v, 16);
        vst1_u16(dst + i, hi);
    }
    for (; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, src + i, 4);
        dst[i] = (uint16_t)(bits >> 16);
    }
}

static inline uint32_t xrand(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
static inline float urand(uint32_t *s) { return (xrand(s) >> 8) * (1.0f / (1u << 24)); }

// Build a 2D row-major NDArray descriptor for BNNS.
static void make_desc(BNNSNDArrayDescriptor *d, size_t rows, size_t cols,
                      void *data, BNNSDataType dtype) {
    memset(d, 0, sizeof(*d));
    d->layout = BNNSDataLayoutRowMajorMatrix;
    // BNNSDataLayoutRowMajorMatrix uses size[0] = cols, size[1] = rows.
    d->size[0] = cols;
    d->size[1] = rows;
    d->stride[0] = 0;
    d->stride[1] = 0;
    d->data = data;
    d->data_type = dtype;
    d->data_scale = 1.0f;
    d->data_bias = 0.0f;
}

// Y[B, R] = X[B, K] @ W_bf[R, K].T  (X bf16, W bf16, Y fp32)
// transB=true means BNNS treats input2 with shape (K, R) instead of (R, K).
static void bf16_matmul(const uint16_t *Wbf, int R, int K,
                        const uint16_t *Xbf, int B,
                        float *Y) {
    BNNSNDArrayDescriptor a, b, c;
    make_desc(&a, B, K, (void *)Xbf, BNNSDataTypeBFloat16);
    make_desc(&b, R, K, (void *)Wbf, BNNSDataTypeBFloat16);
    make_desc(&c, B, R, Y,            BNNSDataTypeFloat32);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    BNNSMatMul(false, true, 1.0f, &a, &b, &c, NULL, NULL);
#pragma clang diagnostic pop
}

static inline void rmsnorm_one(float *x) {
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

static inline int sample_one(const float *p, uint32_t *rng) {
    float r = urand(rng);
    float c = 0.0f;
    for (int i = 0; i < VOCAB - 1; i++) {
        c += p[i];
        if (r < c) return i;
    }
    return VOCAB - 1;
}

static void step_batch(int B, int *toks, int *poses,
                       float *Ks, float *Vs, uint32_t *rngs,
                       float *X, float *XR, float *Q, float *Kbuf, float *Vbuf,
                       float *H, float *HO, float *LG,
                       uint16_t *Xbf, uint16_t *Hbf) {
    for (int b = 0; b < B; b++) {
        const float *wte = W + OFF_WTE + toks[b] * EMBD;
        const float *wpe = W + OFF_WPE + poses[b] * EMBD;
        float *xb = X + b * EMBD;
        for (int i = 0; i < EMBD; i += 4) {
            vst1q_f32(xb + i, vaddq_f32(vld1q_f32(wte + i), vld1q_f32(wpe + i)));
        }
        rmsnorm_one(xb);
    }

    memcpy(XR, X, sizeof(float) * B * EMBD);
    for (int b = 0; b < B; b++) rmsnorm_one(X + b * EMBD);

    f32_to_bf16(X, Xbf, (size_t)B * EMBD);
    bf16_matmul(WQ_bf, EMBD, EMBD, Xbf, B, Q);
    bf16_matmul(WK_bf, EMBD, EMBD, Xbf, B, Kbuf);
    bf16_matmul(WV_bf, EMBD, EMBD, Xbf, B, Vbuf);

    for (int b = 0; b < B; b++) {
        memcpy(Ks + b * BLOCK * EMBD + poses[b] * EMBD, Kbuf + b * EMBD, EMBD * sizeof(float));
        memcpy(Vs + b * BLOCK * EMBD + poses[b] * EMBD, Vbuf + b * EMBD, EMBD * sizeof(float));
    }

    const float scale = 1.0f / 2.0f;
    for (int b = 0; b < B; b++) {
        const float *qb = Q + b * EMBD;
        const float *Kb = Ks + b * BLOCK * EMBD;
        const float *Vb = Vs + b * BLOCK * EMBD;
        float *hob = HO + b * EMBD;
        int t_n = poses[b] + 1;
        for (int hi = 0; hi < HEAD; hi++) {
            const float *qh = qb + hi * HD;
            float al[BLOCK];
            float maxl = -1e30f;
            for (int t = 0; t < t_n; t++) {
                const float *kh = Kb + t * EMBD + hi * HD;
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
                const float *vh = Vb + t * EMBD + hi * HD;
                o0 += w * vh[0]; o1 += w * vh[1]; o2 += w * vh[2]; o3 += w * vh[3];
            }
            hob[hi*HD+0] = o0; hob[hi*HD+1] = o1; hob[hi*HD+2] = o2; hob[hi*HD+3] = o3;
        }
    }

    f32_to_bf16(HO, Xbf, (size_t)B * EMBD);
    bf16_matmul(WO_bf, EMBD, EMBD, Xbf, B, X);
    for (int b = 0; b < B; b++) {
        float *xb = X + b * EMBD;
        const float *xrb = XR + b * EMBD;
        for (int i = 0; i < EMBD; i += 4) {
            vst1q_f32(xb + i, vaddq_f32(vld1q_f32(xb + i), vld1q_f32(xrb + i)));
        }
    }

    memcpy(XR, X, sizeof(float) * B * EMBD);
    for (int b = 0; b < B; b++) rmsnorm_one(X + b * EMBD);
    f32_to_bf16(X, Xbf, (size_t)B * EMBD);
    bf16_matmul(W1_bf, MLP_H, EMBD, Xbf, B, H);
    for (int b = 0; b < B; b++) {
        float *hb = H + b * MLP_H;
        for (int i = 0; i < MLP_H; i += 4) {
            vst1q_f32(hb + i, vmaxq_f32(vld1q_f32(hb + i), vdupq_n_f32(0.0f)));
        }
    }
    f32_to_bf16(H, Hbf, (size_t)B * MLP_H);
    bf16_matmul(W2_bf, EMBD, MLP_H, Hbf, B, X);
    for (int b = 0; b < B; b++) {
        float *xb = X + b * EMBD;
        const float *xrb = XR + b * EMBD;
        for (int i = 0; i < EMBD; i += 4) {
            vst1q_f32(xb + i, vaddq_f32(vld1q_f32(xb + i), vld1q_f32(xrb + i)));
        }
    }

    f32_to_bf16(X, Xbf, (size_t)B * EMBD);
    bf16_matmul(LM_bf, VOCAB, EMBD, Xbf, B, LG);
    for (int b = 0; b < B; b++) {
        float *lg = LG + b * VOCAB;
        float maxl = -1e30f;
        for (int i = 0; i < VOCAB; i++) {
            lg[i] /= TEMP;
            if (lg[i] > maxl) maxl = lg[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < VOCAB; i++) {
            lg[i] = expf(lg[i] - maxl);
            sum += lg[i];
        }
        float inv = 1.0f / sum;
        for (int i = 0; i < VOCAB; i++) lg[i] *= inv;

        int nxt = sample_one(lg, &rngs[b]);
        if (nxt == BOS) {
            toks[b] = BOS; poses[b] = 0;
        } else {
            toks[b] = nxt; poses[b]++;
            if (poses[b] >= BLOCK) { toks[b] = BOS; poses[b] = 0; }
        }
    }
}

static void load_and_quantize_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(W, sizeof(float), TOTAL, f) != TOTAL) {
        fprintf(stderr, "short read\n"); exit(1);
    }
    fclose(f);
    f32_to_bf16(W + OFF_WQ, WQ_bf, EMBD * EMBD);
    f32_to_bf16(W + OFF_WK, WK_bf, EMBD * EMBD);
    f32_to_bf16(W + OFF_WV, WV_bf, EMBD * EMBD);
    f32_to_bf16(W + OFF_WO, WO_bf, EMBD * EMBD);
    f32_to_bf16(W + OFF_W1, W1_bf, MLP_H * EMBD);
    f32_to_bf16(W + OFF_W2, W2_bf, EMBD * MLP_H);
    f32_to_bf16(W + OFF_LM, LM_bf, VOCAB * EMBD);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s BATCH_SIZE [N_STEPS] [WARMUP]\n", argv[0]);
        return 1;
    }
    int B    = atoi(argv[1]);
    long N   = (argc > 2) ? atol(argv[2]) : 100000;
    long WUP = (argc > 3) ? atol(argv[3]) : 5000;
    if (B <= 0) { fprintf(stderr, "BATCH_SIZE must be positive\n"); return 1; }

    load_and_quantize_weights("assets/weights_fp32.bin");

    int *toks      = malloc(sizeof(int) * B);
    int *poses     = malloc(sizeof(int) * B);
    uint32_t *rngs = malloc(sizeof(uint32_t) * B);
    float *Ks      = malloc(sizeof(float) * B * BLOCK * EMBD);
    float *Vs      = malloc(sizeof(float) * B * BLOCK * EMBD);
    float *X       = malloc(sizeof(float) * B * EMBD);
    float *XR      = malloc(sizeof(float) * B * EMBD);
    float *Q       = malloc(sizeof(float) * B * EMBD);
    float *Kbuf    = malloc(sizeof(float) * B * EMBD);
    float *Vbuf    = malloc(sizeof(float) * B * EMBD);
    float *H       = malloc(sizeof(float) * B * MLP_H);
    float *HO      = malloc(sizeof(float) * B * EMBD);
    float *LG      = malloc(sizeof(float) * B * VOCAB);
    uint16_t *Xbf  = malloc(sizeof(uint16_t) * B * EMBD);
    uint16_t *Hbf  = malloc(sizeof(uint16_t) * B * MLP_H);

    memset(Ks, 0, sizeof(float) * B * BLOCK * EMBD);
    memset(Vs, 0, sizeof(float) * B * BLOCK * EMBD);
    for (int b = 0; b < B; b++) {
        toks[b] = BOS; poses[b] = 0; rngs[b] = 42 + b;
    }

    for (long i = 0; i < WUP; i++) {
        step_batch(B, toks, poses, Ks, Vs, rngs, X, XR, Q, Kbuf, Vbuf, H, HO, LG, Xbf, Hbf);
    }
    double t0 = now_sec();
    for (long i = 0; i < N; i++) {
        step_batch(B, toks, poses, Ks, Vs, rngs, X, XR, Q, Kbuf, Vbuf, H, HO, LG, Xbf, Hbf);
    }
    double t1 = now_sec();
    double rate = (double)N * B / (t1 - t0);
    printf("  c BNNS bf16 (batch=%d)        %14.0f tok/sec\n", B, rate);

    free(toks); free(poses); free(rngs);
    free(Ks); free(Vs); free(X); free(XR); free(Q); free(Kbuf); free(Vbuf);
    free(H); free(HO); free(LG); free(Xbf); free(Hbf);
    return 0;
}
