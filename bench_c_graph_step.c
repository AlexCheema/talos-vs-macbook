// microGPT inference via BNNSGraph: one full transformer step compiled
// as a CoreML mlprogram (built by build_coreml_step.py), executed in a
// loop. Weights baked in as fp16. Apple dispatches everything (matmuls,
// rmsnorm, attention, softmax, sample, state update) via the compiled
// graph -- so the per-token cost is one BNNSGraphContextExecute call.
//
// build: clang -O3 -march=native -ffast-math bench_c_graph_step.c -o bench_c_graph_step \
//          -framework Accelerate
// run:   ./bench_c_graph_step BATCH_SIZE [N_STEPS] [WARMUP]
// requires: assets/coreml/step_<BATCH>.mlmodelc (build with build_coreml_step.py + xcrun coremlcompiler)

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#define VOCAB 27
#define BLOCK 16
#define EMBD  16
#define BOS   26

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// xorshift32 → fp16 uniform in [0, 1).
static inline uint32_t xrand(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
static inline uint16_t urand_fp16(uint32_t *s) {
    float v = (xrand(s) >> 8) * (1.0f / (1u << 24));
    float16x4_t h = vcvt_f16_f32(vld1q_dup_f32(&v));
    uint16_t out;
    vst1_lane_u16(&out, vreinterpret_u16_f16(h), 0);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s BATCH_SIZE [N_STEPS] [WARMUP]\n", argv[0]); return 1; }
    int B    = atoi(argv[1]);
    long N   = (argc > 2) ? atol(argv[2]) : 100000;
    long WUP = (argc > 3) ? atol(argv[3]) : 5000;

    char path[256];
    snprintf(path, sizeof(path), "assets/coreml/step_%d.mlmodelc", B);

    bnns_graph_compile_options_t opts = BNNSGraphCompileOptionsMakeDefault();
    BNNSGraphCompileOptionsSetTargetSingleThread(opts, true);
    bnns_graph_t graph = BNNSGraphCompileFromFile(path, NULL, opts);
    BNNSGraphCompileOptionsDestroy(opts);
    if (!graph.data) { fprintf(stderr, "compile failed: %s\n", path); return 1; }

    bnns_graph_context_t ctx = BNNSGraphContextMake(graph);
    if (!ctx.data) { fprintf(stderr, "context make failed\n"); return 1; }

    size_t ws_size = BNNSGraphContextGetWorkspaceSize(ctx, NULL);
    char *ws = NULL;
    if (ws_size != (size_t)-1 && ws_size > 0) {
        if (posix_memalign((void **)&ws, 16384, ws_size) != 0) {
            fprintf(stderr, "ws alloc failed\n"); return 1;
        }
    }

    size_t n_in = BNNSGraphGetInputCount(graph, NULL);
    size_t n_out = BNNSGraphGetOutputCount(graph, NULL);
    fprintf(stderr, "graph: %zu in, %zu out, ws=%zu bytes\n", n_in, n_out, ws_size);

    // Allocate two buffers per state tensor (input + output); we ping-pong
    // each step so the previous output becomes the next input without copy.
    int32_t  *tok[2]; int32_t  *pos[2];
    uint16_t *K[2];   uint16_t *V[2];
    for (int i = 0; i < 2; i++) {
        tok[i] = malloc(sizeof(int32_t) * B);
        pos[i] = malloc(sizeof(int32_t) * B);
        K[i]   = malloc(sizeof(uint16_t) * B * BLOCK * EMBD);
        V[i]   = malloc(sizeof(uint16_t) * B * BLOCK * EMBD);
        memset(K[i], 0, sizeof(uint16_t) * B * BLOCK * EMBD);
        memset(V[i], 0, sizeof(uint16_t) * B * BLOCK * EMBD);
    }
    for (int b = 0; b < B; b++) { tok[0][b] = BOS; pos[0][b] = 0; }
    uint16_t *u_buf = malloc(sizeof(uint16_t) * B);

    uint32_t *rngs = malloc(sizeof(uint32_t) * B);
    for (int b = 0; b < B; b++) rngs[b] = 42 + b;

    // Argument order from build_coreml_step.py: outputs first, then inputs.
    // Outputs: tok_new, pos_new, K_out, V_out
    // Inputs:  tok, pos, K_in, V_in, u
    bnns_graph_argument_t args[9] = {0};

    // Argument order from the compiled graph (queried via inspect tool):
    //   outputs: [0]=tok_new (i32), [1]=pos_new (i32), [2]=K_out (fp16), [3]=V_out (fp16)
    //   inputs:  [4]=K_in, [5]=V_in, [6]=pos, [7]=tok, [8]=u
    int cur = 0;
    #define BIND_AND_EXECUTE() do { \
        int nxt = 1 - cur; \
        for (int b = 0; b < B; b++) u_buf[b] = urand_fp16(&rngs[b]); \
        args[0].data_ptr = tok[nxt]; args[0].data_ptr_size = sizeof(int32_t) * B; \
        args[1].data_ptr = pos[nxt]; args[1].data_ptr_size = sizeof(int32_t) * B; \
        args[2].data_ptr = K[nxt];   args[2].data_ptr_size = sizeof(uint16_t) * B * BLOCK * EMBD; \
        args[3].data_ptr = V[nxt];   args[3].data_ptr_size = sizeof(uint16_t) * B * BLOCK * EMBD; \
        args[4].data_ptr = K[cur];   args[4].data_ptr_size = sizeof(uint16_t) * B * BLOCK * EMBD; \
        args[5].data_ptr = V[cur];   args[5].data_ptr_size = sizeof(uint16_t) * B * BLOCK * EMBD; \
        args[6].data_ptr = pos[cur]; args[6].data_ptr_size = sizeof(int32_t) * B; \
        args[7].data_ptr = tok[cur]; args[7].data_ptr_size = sizeof(int32_t) * B; \
        args[8].data_ptr = u_buf;    args[8].data_ptr_size = sizeof(uint16_t) * B; \
        int rc = BNNSGraphContextExecute(ctx, NULL, 9, args, ws_size, ws); \
        if (rc != 0) { fprintf(stderr, "execute rc=%d\n", rc); exit(1); } \
        cur = nxt; \
    } while (0)

    for (long i = 0; i < WUP; i++) BIND_AND_EXECUTE();
    double t0 = now_sec();
    for (long i = 0; i < N; i++)   BIND_AND_EXECUTE();
    double t1 = now_sec();

    double rate = (double)N * B / (t1 - t0);
    printf("  c BNNSGraph fp16 (batch=%d)        %14.0f tok/sec\n", B, rate);

    for (int i = 0; i < 2; i++) { free(tok[i]); free(pos[i]); free(K[i]); free(V[i]); }
    free(u_buf); free(rngs); free(ws);
    return 0;
}
