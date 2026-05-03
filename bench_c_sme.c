// microGPT batched inference using Apple's Accelerate framework.
// Same structure as bench_c_batch.c but matmuls go through cblas_sgemm,
// which Apple dispatches at runtime to AMX (M1-M3) or SME/SME2 (M4+).
//
// Single binary runs on M3 and M4 — Apple's lib chooses the best path.
//
// build: clang -O3 -march=native -ffast-math bench_c_sme.c -o bench_c_sme \
//          -framework Accelerate
// run:   ./bench_c_sme BATCH_SIZE [N_STEPS] [WARMUP]

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

static inline uint32_t xrand(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
static inline float urand(uint32_t *s) { return (xrand(s) >> 8) * (1.0f / (1u << 24)); }

// Y[B, R] = X[B, K] @ Wm[R, K].T  via Accelerate.
// Apple dispatches to AMX on M3, SME/SME2 on M4.
static inline void mm_AB_T(const float *Wm, const float *X, float *Y,
                           int R, int K, int B) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, R, K,
                1.0f,
                X,  K,
                Wm, K,
                0.0f,
                Y,  R);
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
                       float *H, float *HO, float *LG) {
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

    mm_AB_T(W + OFF_WQ, X, Q,    EMBD, EMBD, B);
    mm_AB_T(W + OFF_WK, X, Kbuf, EMBD, EMBD, B);
    mm_AB_T(W + OFF_WV, X, Vbuf, EMBD, EMBD, B);

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

    mm_AB_T(W + OFF_WO, HO, X, EMBD, EMBD, B);
    for (int b = 0; b < B; b++) {
        float *xb = X + b * EMBD;
        const float *xrb = XR + b * EMBD;
        for (int i = 0; i < EMBD; i += 4) {
            vst1q_f32(xb + i, vaddq_f32(vld1q_f32(xb + i), vld1q_f32(xrb + i)));
        }
    }

    memcpy(XR, X, sizeof(float) * B * EMBD);
    for (int b = 0; b < B; b++) rmsnorm_one(X + b * EMBD);
    mm_AB_T(W + OFF_W1, X, H, MLP_H, EMBD, B);
    for (int b = 0; b < B; b++) {
        float *hb = H + b * MLP_H;
        for (int i = 0; i < MLP_H; i += 4) {
            vst1q_f32(hb + i, vmaxq_f32(vld1q_f32(hb + i), vdupq_n_f32(0.0f)));
        }
    }
    mm_AB_T(W + OFF_W2, H, X, EMBD, MLP_H, B);
    for (int b = 0; b < B; b++) {
        float *xb = X + b * EMBD;
        const float *xrb = XR + b * EMBD;
        for (int i = 0; i < EMBD; i += 4) {
            vst1q_f32(xb + i, vaddq_f32(vld1q_f32(xb + i), vld1q_f32(xrb + i)));
        }
    }

    mm_AB_T(W + OFF_LM, X, LG, VOCAB, EMBD, B);
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

static const char *detect_matrix_path(void) {
    int v = 0;
    size_t sz = sizeof(v);
    if (sysctlbyname("hw.optional.arm.FEAT_SME2", &v, &sz, NULL, 0) == 0 && v) return "SME2";
    if (sysctlbyname("hw.optional.arm.FEAT_SME",  &v, &sz, NULL, 0) == 0 && v) return "SME";
    if (sysctlbyname("hw.optional.amx_version",   &v, &sz, NULL, 0) == 0 && v) return "AMX";
    return "NEON";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s BATCH_SIZE [N_STEPS] [WARMUP]\n", argv[0]);
        return 1;
    }
    int B    = atoi(argv[1]);
    long N   = (argc > 2) ? atol(argv[2]) : 100000;
    long WUP = (argc > 3) ? atol(argv[3]) : 5000;
    int show_label = !(argc > 4 && strcmp(argv[4], "--no-feature") == 0);

    if (B <= 0) { fprintf(stderr, "BATCH_SIZE must be positive\n"); return 1; }

    load_weights("assets/weights_fp32.bin");
    const char *path = detect_matrix_path();

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

    memset(Ks, 0, sizeof(float) * B * BLOCK * EMBD);
    memset(Vs, 0, sizeof(float) * B * BLOCK * EMBD);
    for (int b = 0; b < B; b++) {
        toks[b] = BOS;
        poses[b] = 0;
        rngs[b] = 42 + b;
    }

    for (long i = 0; i < WUP; i++) {
        step_batch(B, toks, poses, Ks, Vs, rngs, X, XR, Q, Kbuf, Vbuf, H, HO, LG);
    }
    double t0 = now_sec();
    for (long i = 0; i < N; i++) {
        step_batch(B, toks, poses, Ks, Vs, rngs, X, XR, Q, Kbuf, Vbuf, H, HO, LG);
    }
    double t1 = now_sec();
    double total_tokens = (double)N * B;
    double rate = total_tokens / (t1 - t0);
    if (show_label) {
        printf("  c Accelerate/%s (batch=%d)        %14.0f tok/sec\n", path, B, rate);
    } else {
        printf("  c Accelerate (batch=%d)        %14.0f tok/sec\n", B, rate);
    }

    free(toks); free(poses); free(rngs);
    free(Ks); free(Vs); free(X); free(XR); free(Q); free(Kbuf); free(Vbuf);
    free(H); free(HO); free(LG);
    return 0;
}
