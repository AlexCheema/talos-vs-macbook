// microGPT batched inference, multi-threaded via Apple Accelerate.
// Splits B streams across T threads. Each thread calls cblas_sgemm on its
// slice -- which on M5 means each thread independently dispatches to the
// SME2 path. Tests whether SME engine can be driven from multiple threads
// in parallel, or whether all threads serialize through one matrix engine.
//
// build: clang -O3 -march=native -ffast-math bench_c_sme_mt.c -o bench_c_sme_mt -framework Accelerate
// run:   ./bench_c_sme_mt BATCH_SIZE N_THREADS [N_STEPS] [WARMUP]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <pthread/qos.h>
#include <arm_neon.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

// Simple counting barrier (macOS pthread has no barrier_t).
typedef struct {
    pthread_mutex_t m;
    pthread_cond_t c;
    int count, n;
    int generation;
} barrier_t;

static void barrier_init(barrier_t *b, int n) {
    pthread_mutex_init(&b->m, NULL);
    pthread_cond_init(&b->c, NULL);
    b->count = 0; b->n = n; b->generation = 0;
}
static void barrier_wait(barrier_t *b) {
    pthread_mutex_lock(&b->m);
    int gen = b->generation;
    if (++b->count == b->n) {
        b->count = 0;
        b->generation++;
        pthread_cond_broadcast(&b->c);
    } else {
        while (b->generation == gen) pthread_cond_wait(&b->c, &b->m);
    }
    pthread_mutex_unlock(&b->m);
}
static void barrier_destroy(barrier_t *b) {
    pthread_mutex_destroy(&b->m);
    pthread_cond_destroy(&b->c);
}

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

// Y[B, R] = X[B, K] @ W[R, K].T  via Accelerate (dispatches to SME2 on M5).
static inline void mm_AB_T(const float *Wm, const float *X, float *Y,
                           int R, int K, int B) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                B, R, K, 1.0f, X, K, Wm, K, 0.0f, Y, R);
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

// Per-thread state.
typedef struct {
    int tid;
    int B_local;
    long N, WUP;
    int *toks, *poses;
    uint32_t *rngs;
    float *Ks, *Vs;
    float *X, *XR, *Q, *Kbuf, *Vbuf, *H, *HO, *LG;
    barrier_t *start_bar, *end_bar;
    double t_start, t_end;  // measured by tid 0 only
} worker_t;

static void *worker(void *arg) {
    worker_t *w = arg;
    // Hint scheduler to put us on P-cores.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);

    for (long i = 0; i < w->WUP; i++) {
        step_batch(w->B_local, w->toks, w->poses, w->Ks, w->Vs, w->rngs,
                   w->X, w->XR, w->Q, w->Kbuf, w->Vbuf, w->H, w->HO, w->LG);
    }

    barrier_wait(w->start_bar);
    if (w->tid == 0) w->t_start = now_sec();

    for (long i = 0; i < w->N; i++) {
        step_batch(w->B_local, w->toks, w->poses, w->Ks, w->Vs, w->rngs,
                   w->X, w->XR, w->Q, w->Kbuf, w->Vbuf, w->H, w->HO, w->LG);
    }

    barrier_wait(w->end_bar);
    if (w->tid == 0) w->t_end = now_sec();
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s BATCH_SIZE N_THREADS [N_STEPS] [WARMUP]\n", argv[0]);
        return 1;
    }
    int B    = atoi(argv[1]);
    int T    = atoi(argv[2]);
    long N   = (argc > 3) ? atol(argv[3]) : 100000;
    long WUP = (argc > 4) ? atol(argv[4]) : 5000;
    if (B <= 0 || T <= 0) { fprintf(stderr, "invalid args\n"); return 1; }
    if (B % T != 0) { fprintf(stderr, "BATCH_SIZE must be a multiple of N_THREADS\n"); return 1; }
    int B_local = B / T;

    load_weights("assets/weights_fp32.bin");

    barrier_t start_bar, end_bar;
    barrier_init(&start_bar, T);
    barrier_init(&end_bar,   T);

    worker_t *workers = calloc(T, sizeof(worker_t));
    pthread_t *thrs = calloc(T, sizeof(pthread_t));
    for (int t = 0; t < T; t++) {
        worker_t *w = &workers[t];
        w->tid = t; w->B_local = B_local; w->N = N; w->WUP = WUP;
        w->start_bar = &start_bar; w->end_bar = &end_bar;
        w->toks = malloc(sizeof(int) * B_local);
        w->poses = malloc(sizeof(int) * B_local);
        w->rngs = malloc(sizeof(uint32_t) * B_local);
        w->Ks = malloc(sizeof(float) * B_local * BLOCK * EMBD);
        w->Vs = malloc(sizeof(float) * B_local * BLOCK * EMBD);
        w->X = malloc(sizeof(float) * B_local * EMBD);
        w->XR = malloc(sizeof(float) * B_local * EMBD);
        w->Q = malloc(sizeof(float) * B_local * EMBD);
        w->Kbuf = malloc(sizeof(float) * B_local * EMBD);
        w->Vbuf = malloc(sizeof(float) * B_local * EMBD);
        w->H = malloc(sizeof(float) * B_local * MLP_H);
        w->HO = malloc(sizeof(float) * B_local * EMBD);
        w->LG = malloc(sizeof(float) * B_local * VOCAB);
        memset(w->Ks, 0, sizeof(float) * B_local * BLOCK * EMBD);
        memset(w->Vs, 0, sizeof(float) * B_local * BLOCK * EMBD);
        for (int b = 0; b < B_local; b++) {
            w->toks[b] = BOS; w->poses[b] = 0;
            w->rngs[b] = 42 + t * B_local + b;
        }
    }

    for (int t = 0; t < T; t++) pthread_create(&thrs[t], NULL, worker, &workers[t]);
    for (int t = 0; t < T; t++) pthread_join(thrs[t], NULL);

    double total_tokens = (double)N * B;
    double rate = total_tokens / (workers[0].t_end - workers[0].t_start);
    printf("  c Accelerate/SME2 (batch=%d t=%d)        %14.0f tok/sec\n", B, T, rate);

    for (int t = 0; t < T; t++) {
        worker_t *w = &workers[t];
        free(w->toks); free(w->poses); free(w->rngs);
        free(w->Ks); free(w->Vs); free(w->X); free(w->XR); free(w->Q);
        free(w->Kbuf); free(w->Vbuf); free(w->H); free(w->HO); free(w->LG);
    }
    free(workers); free(thrs);
    barrier_destroy(&start_bar); barrier_destroy(&end_bar);
    return 0;
}
