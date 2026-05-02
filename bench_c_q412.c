// microGPT inference, single-thread, Q4.12 fixed-point matmuls (matches TALOS-V2 arithmetic).
// Weights are int16 (round(fp * 4096)). Activations carried as int32 in Q4.12.
// RMSNorm/softmax done in float for simplicity (TALOS uses LUT/Newton in hardware).
//
// build: clang -O3 -march=native -ffast-math bench_c_q412.c -o bench_c_q412

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
#define Q 12
#define SCALE 4096.0f

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

static int16_t W[TOTAL];

static uint32_t rng_state = 42;
static inline uint32_t xrand(void) {
    uint32_t x = rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    rng_state = x;
    return x;
}
static inline float urand(void) { return (xrand() >> 8) * (1.0f / (1u << 24)); }

// y[R] = (W[R,16] @ x[16]) >> Q   in Q4.12. NEON widening MAC.
static inline void matvec_16in_q(const int16_t *Wm, const int16_t *x, int16_t *y, int R) {
    int16x8_t x0 = vld1q_s16(x);
    int16x8_t x1 = vld1q_s16(x + 8);
    for (int r = 0; r < R; r++) {
        const int16_t *wr = Wm + r * EMBD;
        int16x8_t w0 = vld1q_s16(wr);
        int16x8_t w1 = vld1q_s16(wr + 8);
        int32x4_t a = vmull_s16(vget_low_s16(w0), vget_low_s16(x0));
        a = vmlal_s16(a, vget_high_s16(w0), vget_high_s16(x0));
        a = vmlal_s16(a, vget_low_s16(w1), vget_low_s16(x1));
        a = vmlal_s16(a, vget_high_s16(w1), vget_high_s16(x1));
        int32_t s = vaddvq_s32(a);
        s >>= Q;
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        y[r] = (int16_t)s;
    }
}

// y[16] = (W[16,64] @ x[64]) >> Q
static inline void matvec_mlp_out_q(const int16_t *Wm, const int16_t *x, int16_t *y) {
    for (int r = 0; r < EMBD; r++) {
        const int16_t *wr = Wm + r * MLP_H;
        int32x4_t acc = vdupq_n_s32(0);
        for (int c = 0; c < MLP_H; c += 16) {
            int16x8_t w0 = vld1q_s16(wr + c);
            int16x8_t w1 = vld1q_s16(wr + c + 8);
            int16x8_t x0 = vld1q_s16(x + c);
            int16x8_t x1 = vld1q_s16(x + c + 8);
            acc = vmlal_s16(acc, vget_low_s16(w0), vget_low_s16(x0));
            acc = vmlal_s16(acc, vget_high_s16(w0), vget_high_s16(x0));
            acc = vmlal_s16(acc, vget_low_s16(w1), vget_low_s16(x1));
            acc = vmlal_s16(acc, vget_high_s16(w1), vget_high_s16(x1));
        }
        int32_t s = vaddvq_s32(acc);
        s >>= Q;
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        y[r] = (int16_t)s;
    }
}

static inline void rmsnorm_q(int16_t *x) {
    // Convert to float, compute RMSNorm, convert back.
    float fx[EMBD];
    float ms = 0.0f;
    for (int i = 0; i < EMBD; i++) { fx[i] = x[i] / SCALE; ms += fx[i] * fx[i]; }
    float scale = 1.0f / sqrtf(ms / EMBD + 1e-5f);
    for (int i = 0; i < EMBD; i++) {
        float v = fx[i] * scale * SCALE;
        int s = (int)lrintf(v);
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        x[i] = (int16_t)s;
    }
}

static inline int sample_probs(const float *p) {
    float r = urand();
    float c = 0.0f;
    for (int i = 0; i < VOCAB - 1; i++) {
        c += p[i];
        if (r < c) return i;
    }
    return VOCAB - 1;
}

static inline int step(int tok, int pos, int16_t *K, int16_t *V) {
    int16_t x[EMBD] __attribute__((aligned(16)));
    int16_t xr[EMBD] __attribute__((aligned(16)));
    int16_t q[EMBD] __attribute__((aligned(16)));
    int16_t k[EMBD] __attribute__((aligned(16)));
    int16_t v[EMBD] __attribute__((aligned(16)));
    int16_t h[MLP_H] __attribute__((aligned(16)));
    int16_t head_out[EMBD] __attribute__((aligned(16)));
    int16_t lm[VOCAB];

    const int16_t *wte = W + OFF_WTE + tok * EMBD;
    const int16_t *wpe = W + OFF_WPE + pos * EMBD;
    for (int i = 0; i < EMBD; i++) x[i] = wte[i] + wpe[i]; // both Q4.12, sum stays Q4.12
    rmsnorm_q(x);

    memcpy(xr, x, sizeof(x));
    rmsnorm_q(x);

    matvec_16in_q(W + OFF_WQ, x, q, EMBD);
    matvec_16in_q(W + OFF_WK, x, k, EMBD);
    matvec_16in_q(W + OFF_WV, x, v, EMBD);

    memcpy(K + pos * EMBD, k, sizeof(k));
    memcpy(V + pos * EMBD, v, sizeof(v));

    // Attention: do in float for numerical sanity (still Q4.12 inputs).
    int t_n = pos + 1;
    float fhead[EMBD];
    const float scale = 1.0f / 2.0f / SCALE;  // 1/sqrt(HD) and Q->float of one operand
    for (int hi = 0; hi < HEAD; hi++) {
        int16_t *qh = q + hi * HD;
        float al[BLOCK];
        float maxl = -1e30f;
        for (int t = 0; t < t_n; t++) {
            const int16_t *kh = K + t * EMBD + hi * HD;
            int32_t dot = qh[0]*kh[0] + qh[1]*kh[1] + qh[2]*kh[2] + qh[3]*kh[3];
            // dot is in Q8.24. Convert to float and apply 1/sqrt(HD).
            al[t] = (dot / SCALE / SCALE) * 0.5f;
            if (al[t] > maxl) maxl = al[t];
        }
        float sum = 0.0f;
        for (int t = 0; t < t_n; t++) { al[t] = expf(al[t] - maxl); sum += al[t]; }
        float inv = 1.0f / sum;
        float o0=0, o1=0, o2=0, o3=0;
        for (int t = 0; t < t_n; t++) {
            float w = al[t] * inv;
            const int16_t *vh = V + t * EMBD + hi * HD;
            o0 += w * (vh[0] / SCALE);
            o1 += w * (vh[1] / SCALE);
            o2 += w * (vh[2] / SCALE);
            o3 += w * (vh[3] / SCALE);
        }
        fhead[hi*HD+0]=o0; fhead[hi*HD+1]=o1; fhead[hi*HD+2]=o2; fhead[hi*HD+3]=o3;
    }
    for (int i = 0; i < EMBD; i++) {
        int s = (int)lrintf(fhead[i] * SCALE);
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        head_out[i] = (int16_t)s;
    }
    (void)scale;

    matvec_16in_q(W + OFF_WO, head_out, x, EMBD);
    for (int i = 0; i < EMBD; i++) {
        int s = (int)x[i] + (int)xr[i];
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        x[i] = (int16_t)s;
    }

    memcpy(xr, x, sizeof(x));
    rmsnorm_q(x);

    matvec_16in_q(W + OFF_W1, x, h, MLP_H);
    for (int i = 0; i < MLP_H; i++) if (h[i] < 0) h[i] = 0;
    matvec_mlp_out_q(W + OFF_W2, h, x);
    for (int i = 0; i < EMBD; i++) {
        int s = (int)x[i] + (int)xr[i];
        if (s > 32767) s = 32767; else if (s < -32768) s = -32768;
        x[i] = (int16_t)s;
    }

    matvec_16in_q(W + OFF_LM, x, lm, VOCAB);
    float logits[VOCAB];
    float maxl = -1e30f;
    for (int i = 0; i < VOCAB; i++) {
        logits[i] = (lm[i] / SCALE) / TEMP;
        if (logits[i] > maxl) maxl = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < VOCAB; i++) { logits[i] = expf(logits[i] - maxl); sum += logits[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < VOCAB; i++) logits[i] *= inv;

    return sample_probs(logits);
}

static void load_weights(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(W, sizeof(int16_t), TOTAL, f) != TOTAL) {
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

    load_weights("assets/weights_q412.bin");

    if (names_mode) {
        const char chars[] = "abcdefghijklmnopqrstuvwxyz";
        for (int s = 0; s < 20; s++) {
            int16_t K[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
            int16_t V[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
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

    int16_t K[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
    int16_t V[BLOCK * EMBD] __attribute__((aligned(16))) = {0};
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
    printf("  c Q4.12 fixed-point      %14.0f tok/sec\n", rate);
    return 0;
}
