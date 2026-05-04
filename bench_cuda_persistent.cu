// microGPT inference, single-stream, batch=1, persistent CUDA kernel.
// Weights laid out per WEIGHT_ORDER in model.py.
//
// One kernel launch covers the entire timed run. Weights live in shared
// memory; the per-token forward pass and sampler are cooperative across
// a single warp. No host round-trips while timing.
//
// build: nvcc -O3 -arch=sm_121 bench_cuda_persistent.cu -o bench_cuda_persistent
// run:   ./bench_cuda_persistent [N_TOKENS] [WARMUP]

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

#define TPB 32  // threads per block; one warp

#define CK(x) do { cudaError_t e = (x); if (e) { \
    fprintf(stderr, "cuda: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

// Per-block shared state. Weights, KV cache, activations, RNG.
struct State {
    float W[TOTAL];
    float K[BLOCK * EMBD];
    float V[BLOCK * EMBD];
    float x[EMBD];
    float xr[EMBD];
    float q[EMBD];
    float kbuf[EMBD];
    float vbuf[EMBD];
    float head_out[EMBD];
    float h[MLP_H];
    float logits[VOCAB];
    uint32_t rng;
};

// Warp-wide RMSNorm over the first EMBD lanes; remaining lanes contribute 0.
__device__ inline void rmsnorm(float *x, int tid) {
    float xi = (tid < EMBD) ? x[tid] : 0.f;
    float ss = xi * xi;
    ss += __shfl_xor_sync(0xffffffff, ss, 16);
    ss += __shfl_xor_sync(0xffffffff, ss, 8);
    ss += __shfl_xor_sync(0xffffffff, ss, 4);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    float scale = 1.0f / sqrtf(ss / EMBD + 1e-5f);
    if (tid < EMBD) x[tid] = xi * scale;
}

// y = Wm @ x where Wm is (R, EMBD) row-major. Strided over the warp.
__device__ inline void matvec_embd(const float *Wm, const float *x, float *y, int R, int tid) {
    for (int r = tid; r < R; r += TPB) {
        const float *wr = Wm + r * EMBD;
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < EMBD; i++) a += wr[i] * x[i];
        y[r] = a;
    }
}

// y = Wm @ h where Wm is (EMBD, MLP_H) row-major.
__device__ inline void matvec_mlp_out(const float *Wm, const float *h, float *y, int tid) {
    if (tid < EMBD) {
        const float *wr = Wm + tid * MLP_H;
        float a = 0.f;
        #pragma unroll
        for (int i = 0; i < MLP_H; i++) a += wr[i] * h[i];
        y[tid] = a;
    }
}

__device__ inline int sample_probs(const float *p, uint32_t *rng) {
    uint32_t x = *rng;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *rng = x;
    float r = (x >> 8) * (1.0f / (1u << 24));
    float c = 0.f;
    for (int i = 0; i < VOCAB - 1; i++) {
        c += p[i];
        if (r < c) return i;
    }
    return VOCAB - 1;
}

// One forward pass + sampled token. Writes through s->logits; returns next tok.
__device__ inline int forward(State *s, int tok, int pos, int tid) {
    if (tid < EMBD) s->x[tid] = s->W[OFF_WTE + tok * EMBD + tid] +
                                s->W[OFF_WPE + pos * EMBD + tid];
    __syncwarp();
    rmsnorm(s->x, tid);

    if (tid < EMBD) s->xr[tid] = s->x[tid];
    __syncwarp();
    rmsnorm(s->x, tid);

    matvec_embd(s->W + OFF_WQ, s->x, s->q,    EMBD, tid);
    matvec_embd(s->W + OFF_WK, s->x, s->kbuf, EMBD, tid);
    matvec_embd(s->W + OFF_WV, s->x, s->vbuf, EMBD, tid);
    __syncwarp();

    if (tid < EMBD) {
        s->K[pos * EMBD + tid] = s->kbuf[tid];
        s->V[pos * EMBD + tid] = s->vbuf[tid];
    }
    __syncwarp();

    if (tid < HEAD) {
        const float *qh = s->q + tid * HD;
        float al[BLOCK];
        float maxl = -1e30f;
        const float scale = 0.5f; // 1/sqrt(HD=4)
        int t_n = pos + 1;
        for (int t = 0; t < t_n; t++) {
            const float *kh = s->K + t * EMBD + tid * HD;
            float dot = qh[0]*kh[0] + qh[1]*kh[1] + qh[2]*kh[2] + qh[3]*kh[3];
            al[t] = dot * scale;
            if (al[t] > maxl) maxl = al[t];
        }
        float sum = 0.f;
        for (int t = 0; t < t_n; t++) { al[t] = expf(al[t] - maxl); sum += al[t]; }
        float inv = 1.f / sum;
        float o0=0,o1=0,o2=0,o3=0;
        for (int t = 0; t < t_n; t++) {
            float w = al[t] * inv;
            const float *vh = s->V + t * EMBD + tid * HD;
            o0 += w * vh[0]; o1 += w * vh[1]; o2 += w * vh[2]; o3 += w * vh[3];
        }
        s->head_out[tid*HD+0]=o0; s->head_out[tid*HD+1]=o1;
        s->head_out[tid*HD+2]=o2; s->head_out[tid*HD+3]=o3;
    }
    __syncwarp();

    matvec_embd(s->W + OFF_WO, s->head_out, s->x, EMBD, tid);
    __syncwarp();
    if (tid < EMBD) s->x[tid] += s->xr[tid];
    if (tid < EMBD) s->xr[tid] = s->x[tid];
    __syncwarp();
    rmsnorm(s->x, tid);

    matvec_embd(s->W + OFF_W1, s->x, s->h, MLP_H, tid);
    __syncwarp();
    for (int r = tid; r < MLP_H; r += TPB) {
        float v = s->h[r];
        s->h[r] = v > 0.f ? v : 0.f;
    }
    __syncwarp();
    matvec_mlp_out(s->W + OFF_W2, s->h, s->x, tid);
    __syncwarp();
    if (tid < EMBD) s->x[tid] += s->xr[tid];
    __syncwarp();

    matvec_embd(s->W + OFF_LM, s->x, s->logits, VOCAB, tid);
    __syncwarp();

    if (tid == 0) {
        float maxl = -1e30f;
        for (int i = 0; i < VOCAB; i++) {
            s->logits[i] /= TEMP;
            if (s->logits[i] > maxl) maxl = s->logits[i];
        }
        float sum = 0.f;
        for (int i = 0; i < VOCAB; i++) { s->logits[i] = expf(s->logits[i] - maxl); sum += s->logits[i]; }
        float inv = 1.f / sum;
        for (int i = 0; i < VOCAB; i++) s->logits[i] *= inv;
    }
    __syncwarp();

    int next_tok = (tid == 0) ? sample_probs(s->logits, &s->rng) : 0;
    next_tok = __shfl_sync(0xffffffff, next_tok, 0);
    return next_tok;
}

__device__ inline void load_weights(State *s, const float *Wg, int tid) {
    for (int i = tid; i < TOTAL; i += TPB) s->W[i] = Wg[i];
}

__device__ inline void clear_kv(State *s, int tid) {
    for (int i = tid; i < BLOCK * EMBD; i += TPB) { s->K[i] = 0.f; s->V[i] = 0.f; }
}

__global__ void k_run(const float *Wg, int n_tokens, uint32_t rng_init) {
    __shared__ State s;
    int tid = threadIdx.x;
    load_weights(&s, Wg, tid);
    clear_kv(&s, tid);
    if (tid == 0) s.rng = rng_init;
    __syncthreads();

    int tok = BOS, pos = 0;
    for (int step = 0; step < n_tokens; step++) {
        if (pos >= BLOCK) { tok = BOS; pos = 0; clear_kv(&s, tid); __syncwarp(); }
        int nxt = forward(&s, tok, pos, tid);
        if (nxt == BOS) { tok = BOS; pos = 0; }
        else            { tok = nxt; pos++; }
    }
}

__global__ void k_names(const float *Wg, uint32_t rng_init,
                        int *tokens_out, int *lens_out) {
    __shared__ State s;
    int tid = threadIdx.x;
    load_weights(&s, Wg, tid);
    if (tid == 0) s.rng = rng_init;
    __syncthreads();

    for (int sn = 0; sn < 20; sn++) {
        clear_kv(&s, tid);
        __syncwarp();
        int tok = BOS;
        int len = 0;
        for (int pos = 0; pos < BLOCK; pos++) {
            int nxt = forward(&s, tok, pos, tid);
            if (nxt == BOS) break;
            if (tid == 0) tokens_out[sn * BLOCK + len] = nxt;
            len++;
            tok = nxt;
        }
        if (tid == 0) lens_out[sn] = len;
        __syncwarp();
    }
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void load_weights_host(const char *path, float *host_W) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    if (fread(host_W, sizeof(float), TOTAL, f) != TOTAL) {
        fprintf(stderr, "short read\n"); exit(1);
    }
    fclose(f);
}

int main(int argc, char **argv) {
    int names_mode = (argc > 1 && strcmp(argv[1], "--names") == 0);
    long N   = (!names_mode && argc > 1) ? atol(argv[1]) : 1000000;
    long WUP = (argc > 2) ? atol(argv[2]) : 50000;

    float *host_W = (float*)malloc(TOTAL * sizeof(float));
    load_weights_host("assets/weights_fp32.bin", host_W);

    float *Wd;
    CK(cudaMalloc(&Wd, TOTAL * sizeof(float)));
    CK(cudaMemcpy(Wd, host_W, TOTAL * sizeof(float), cudaMemcpyHostToDevice));
    free(host_W);

    cudaFuncSetAttribute(k_run,   cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
    cudaFuncSetAttribute(k_names, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);

    if (names_mode) {
        int *toks_d, *lens_d;
        CK(cudaMalloc(&toks_d, 20 * BLOCK * sizeof(int)));
        CK(cudaMalloc(&lens_d, 20 * sizeof(int)));
        k_names<<<1, TPB>>>(Wd, 42u, toks_d, lens_d);
        CK(cudaDeviceSynchronize());

        int toks[20 * BLOCK], lens[20];
        CK(cudaMemcpy(toks, toks_d, sizeof(toks), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(lens, lens_d, sizeof(lens), cudaMemcpyDeviceToHost));
        const char chars[] = "abcdefghijklmnopqrstuvwxyz";
        for (int s = 0; s < 20; s++) {
            char buf[BLOCK + 1] = {0};
            for (int i = 0; i < lens[s]; i++) buf[i] = chars[toks[s * BLOCK + i]];
            printf("sample %2d: %s\n", s + 1, buf);
        }
        return 0;
    }

    k_run<<<1, TPB>>>(Wd, (int)WUP, 42u);
    CK(cudaDeviceSynchronize());

    double t0 = now_sec();
    k_run<<<1, TPB>>>(Wd, (int)N, 42u);
    CK(cudaDeviceSynchronize());
    double t1 = now_sec();

    double rate = N / (t1 - t0);
    printf("  cuda persistent          %14.0f tok/sec\n", rate);
    return 0;
}
