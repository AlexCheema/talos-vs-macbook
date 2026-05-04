// microGPT inference, single-stream, batch=1, naive launch-per-op CUDA.
// Weights laid out per WEIGHT_ORDER in model.py.
//
// build: nvcc -O3 -arch=sm_121 bench_cuda.cu -o bench_cuda
// run:   ./bench_cuda [N_TOKENS] [WARMUP]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

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
#define OFF_WPE (OFF_WTE + VOCAB * EMBD)
#define OFF_WQ  (OFF_WPE + BLOCK * EMBD)
#define OFF_WK  (OFF_WQ  + EMBD * EMBD)
#define OFF_WV  (OFF_WK  + EMBD * EMBD)
#define OFF_WO  (OFF_WV  + EMBD * EMBD)
#define OFF_W1  (OFF_WO  + EMBD * EMBD)
#define OFF_W2  (OFF_W1  + MLP_H * EMBD)
#define OFF_LM  (OFF_W2  + EMBD * MLP_H)
#define TOTAL   (OFF_LM  + VOCAB * EMBD)

#define CK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

// y[i] = wte[tok,i] + wpe[pos,i].  EMBD threads.
__global__ void k_embed(const float *W, const int *tok_dev, int pos, float *y) {
    int i = threadIdx.x;
    int tok = *tok_dev;
    y[i] = W[OFF_WTE + tok * EMBD + i] + W[OFF_WPE + pos * EMBD + i];
}

// RMSNorm over EMBD in-place. Single warp, EMBD=16 lanes.
__global__ void k_rmsnorm(float *x) {
    __shared__ float scale;
    int i = threadIdx.x;
    float xi = x[i];
    float ss = xi * xi;
    for (int o = 8; o; o >>= 1) ss += __shfl_xor_sync(0xffff, ss, o);
    if (i == 0) scale = rsqrtf(ss / EMBD + 1e-5f);
    __syncthreads();
    x[i] = xi * scale;
}

__global__ void k_copy_embd(float *dst, const float *src) {
    dst[threadIdx.x] = src[threadIdx.x];
}

__global__ void k_add_embd(float *y, const float *x) {
    y[threadIdx.x] += x[threadIdx.x];
}

// y = Wm @ x where Wm is (R, EMBD) row-major. One thread per output row.
__global__ void k_matvec_embd(const float *Wm, const float *x, float *y, int R) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    const float *wr = Wm + r * EMBD;
    float a = 0.f;
    #pragma unroll
    for (int i = 0; i < EMBD; i++) a += wr[i] * x[i];
    y[r] = a;
}

// y = Wm @ h where Wm is (EMBD, MLP_H) row-major.
__global__ void k_matvec_mlp_out(const float *Wm, const float *h, float *y) {
    int r = threadIdx.x;
    const float *wr = Wm + r * MLP_H;
    float a = 0.f;
    #pragma unroll
    for (int i = 0; i < MLP_H; i++) a += wr[i] * h[i];
    y[r] = a;
}

__global__ void k_relu_mlp(float *h) {
    float v = h[threadIdx.x];
    h[threadIdx.x] = v > 0.f ? v : 0.f;
}

__global__ void k_kv_write(float *K, float *V, const float *k, const float *v, int pos) {
    int i = threadIdx.x;
    K[pos * EMBD + i] = k[i];
    V[pos * EMBD + i] = v[i];
}

// Per-head softmax-attention, 4 heads × HD=4. One thread per head.
__global__ void k_attention(const float *K, const float *V, const float *q,
                            float *head_out, int t_n) {
    int hi = threadIdx.x;
    const float *qh = q + hi * HD;
    float al[BLOCK];
    float maxl = -1e30f;
    const float scale = 0.5f; // 1/sqrt(HD=4)
    for (int t = 0; t < t_n; t++) {
        const float *kh = K + t * EMBD + hi * HD;
        float dot = qh[0]*kh[0] + qh[1]*kh[1] + qh[2]*kh[2] + qh[3]*kh[3];
        al[t] = dot * scale;
        if (al[t] > maxl) maxl = al[t];
    }
    float sum = 0.f;
    for (int t = 0; t < t_n; t++) { al[t] = __expf(al[t] - maxl); sum += al[t]; }
    float inv = 1.f / sum;
    float o0=0, o1=0, o2=0, o3=0;
    for (int t = 0; t < t_n; t++) {
        float w = al[t] * inv;
        const float *vh = V + t * EMBD + hi * HD;
        o0 += w * vh[0]; o1 += w * vh[1]; o2 += w * vh[2]; o3 += w * vh[3];
    }
    head_out[hi*HD+0]=o0; head_out[hi*HD+1]=o1;
    head_out[hi*HD+2]=o2; head_out[hi*HD+3]=o3;
}

// Softmax(logits / TEMP) in place. One thread per vocab entry.
__global__ void k_softmax_temp(float *logits) {
    __shared__ float sh[VOCAB];
    __shared__ float maxl, denom;
    int i = threadIdx.x;
    float v = logits[i] / TEMP;
    sh[i] = v;
    __syncthreads();
    if (i == 0) {
        float m = -1e30f;
        for (int j = 0; j < VOCAB; j++) if (sh[j] > m) m = sh[j];
        maxl = m;
    }
    __syncthreads();
    v = __expf(v - maxl);
    sh[i] = v;
    __syncthreads();
    if (i == 0) {
        float s = 0.f;
        for (int j = 0; j < VOCAB; j++) s += sh[j];
        denom = s;
    }
    __syncthreads();
    logits[i] = v / denom;
}

// xorshift32 + cumulative-scan sample.
__global__ void k_sample(const float *p, uint32_t *rng, int *tok_out) {
    uint32_t x = *rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *rng = x;
    float r = (x >> 8) * (1.0f / (1u << 24));
    float c = 0.f;
    int chosen = VOCAB - 1;
    for (int i = 0; i < VOCAB - 1; i++) {
        c += p[i];
        if (r < c) { chosen = i; break; }
    }
    *tok_out = chosen;
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void load_weights(const char *path, float *host_W) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(host_W, sizeof(float), TOTAL, f) != TOTAL) {
        fprintf(stderr, "short read\n"); exit(1);
    }
    fclose(f);
}

// One forward pass: tok_dev[in] -> tok_dev[out].
static inline void step(float *Wd, float *Kd, float *Vd,
                        float *xd, float *xrd,
                        float *qd, float *kd, float *vd,
                        float *hd, float *headd, float *logitsd,
                        int *tok_dev, uint32_t *rng_dev, int pos) {
    k_embed<<<1, EMBD>>>(Wd, tok_dev, pos, xd);
    k_rmsnorm<<<1, EMBD>>>(xd);
    k_copy_embd<<<1, EMBD>>>(xrd, xd);
    k_rmsnorm<<<1, EMBD>>>(xd);

    k_matvec_embd<<<1, EMBD>>>(Wd + OFF_WQ, xd, qd, EMBD);
    k_matvec_embd<<<1, EMBD>>>(Wd + OFF_WK, xd, kd, EMBD);
    k_matvec_embd<<<1, EMBD>>>(Wd + OFF_WV, xd, vd, EMBD);

    k_kv_write<<<1, EMBD>>>(Kd, Vd, kd, vd, pos);
    k_attention<<<1, HEAD>>>(Kd, Vd, qd, headd, pos + 1);

    k_matvec_embd<<<1, EMBD>>>(Wd + OFF_WO, headd, xd, EMBD);
    k_add_embd<<<1, EMBD>>>(xd, xrd);

    k_copy_embd<<<1, EMBD>>>(xrd, xd);
    k_rmsnorm<<<1, EMBD>>>(xd);

    k_matvec_embd<<<1, MLP_H>>>(Wd + OFF_W1, xd, hd, MLP_H);
    k_relu_mlp<<<1, MLP_H>>>(hd);
    k_matvec_mlp_out<<<1, EMBD>>>(Wd + OFF_W2, hd, xd);
    k_add_embd<<<1, EMBD>>>(xd, xrd);

    k_matvec_embd<<<1, VOCAB>>>(Wd + OFF_LM, xd, logitsd, VOCAB);
    k_softmax_temp<<<1, VOCAB>>>(logitsd);
    k_sample<<<1, 1>>>(logitsd, rng_dev, tok_dev);
}

int main(int argc, char **argv) {
    int names_mode = (argc > 1 && strcmp(argv[1], "--names") == 0);
    long N   = (!names_mode && argc > 1) ? atol(argv[1]) : 1000000;
    long WUP = (argc > 2) ? atol(argv[2]) : 50000;

    float *host_W = (float*)malloc(TOTAL * sizeof(float));
    load_weights("assets/weights_fp32.bin", host_W);

    float *Wd, *Kd, *Vd, *xd, *xrd, *qd, *kd, *vd, *hd, *headd, *logitsd;
    int *tok_dev;
    uint32_t *rng_dev;
    CK(cudaMalloc(&Wd,      TOTAL * sizeof(float)));
    CK(cudaMalloc(&Kd,      BLOCK * EMBD * sizeof(float)));
    CK(cudaMalloc(&Vd,      BLOCK * EMBD * sizeof(float)));
    CK(cudaMalloc(&xd,      EMBD * sizeof(float)));
    CK(cudaMalloc(&xrd,     EMBD * sizeof(float)));
    CK(cudaMalloc(&qd,      EMBD * sizeof(float)));
    CK(cudaMalloc(&kd,      EMBD * sizeof(float)));
    CK(cudaMalloc(&vd,      EMBD * sizeof(float)));
    CK(cudaMalloc(&hd,      MLP_H * sizeof(float)));
    CK(cudaMalloc(&headd,   EMBD * sizeof(float)));
    CK(cudaMalloc(&logitsd, VOCAB * sizeof(float)));
    CK(cudaMalloc(&tok_dev, sizeof(int)));
    CK(cudaMalloc(&rng_dev, sizeof(uint32_t)));

    CK(cudaMemcpy(Wd, host_W, TOTAL * sizeof(float), cudaMemcpyHostToDevice));
    free(host_W);
    uint32_t rng_init = 42;
    CK(cudaMemcpy(rng_dev, &rng_init, sizeof(uint32_t), cudaMemcpyHostToDevice));

    if (names_mode) {
        const char chars[] = "abcdefghijklmnopqrstuvwxyz";
        for (int s = 0; s < 20; s++) {
            CK(cudaMemset(Kd, 0, BLOCK * EMBD * sizeof(float)));
            CK(cudaMemset(Vd, 0, BLOCK * EMBD * sizeof(float)));
            int tok = BOS;
            CK(cudaMemcpy(tok_dev, &tok, sizeof(int), cudaMemcpyHostToDevice));
            char buf[BLOCK + 1] = {0};
            int len = 0;
            for (int pos = 0; pos < BLOCK; pos++) {
                step(Wd, Kd, Vd, xd, xrd, qd, kd, vd, hd, headd, logitsd,
                           tok_dev, rng_dev, pos);
                CK(cudaMemcpy(&tok, tok_dev, sizeof(int), cudaMemcpyDeviceToHost));
                if (tok == BOS) break;
                buf[len++] = chars[tok];
            }
            printf("sample %2d: %s\n", s + 1, buf);
        }
        return 0;
    }

    int tok = BOS, pos = 0;
    CK(cudaMemcpy(tok_dev, &tok, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemset(Kd, 0, BLOCK * EMBD * sizeof(float)));
    CK(cudaMemset(Vd, 0, BLOCK * EMBD * sizeof(float)));

    for (long i = 0; i < WUP; i++) {
        if (pos >= BLOCK) {
            tok = BOS; pos = 0;
            CK(cudaMemcpy(tok_dev, &tok, sizeof(int), cudaMemcpyHostToDevice));
        }
        step(Wd, Kd, Vd, xd, xrd, qd, kd, vd, hd, headd, logitsd,
                   tok_dev, rng_dev, pos);
        CK(cudaMemcpy(&tok, tok_dev, sizeof(int), cudaMemcpyDeviceToHost));
        if (tok == BOS) { pos = 0; }
        else            { pos++; }
    }
    CK(cudaDeviceSynchronize());

    double t0 = now_sec();
    long emitted = 0;
    for (long i = 0; i < N; i++) {
        if (pos >= BLOCK) {
            tok = BOS; pos = 0;
            CK(cudaMemcpy(tok_dev, &tok, sizeof(int), cudaMemcpyHostToDevice));
        }
        step(Wd, Kd, Vd, xd, xrd, qd, kd, vd, hd, headd, logitsd,
                   tok_dev, rng_dev, pos);
        CK(cudaMemcpy(&tok, tok_dev, sizeof(int), cudaMemcpyDeviceToHost));
        emitted++;
        if (tok == BOS) { pos = 0; }
        else            { pos++; }
    }
    CK(cudaDeviceSynchronize());
    double t1 = now_sec();
    double rate = emitted / (t1 - t0);
    printf("  cuda fp32                %14.0f tok/sec\n", rate);
    return 0;
}
