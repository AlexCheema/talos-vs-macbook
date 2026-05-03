// Probe: does BNNSGraph have lower per-call overhead than BNNSMatMul?
// Loads matmul_512.mlmodelc (single bf16 matmul, B=512 baked in) and
// times N executions back to back.
//
// build: clang -O3 -march=native -ffast-math bench_c_graph_probe.c -o bench_c_graph_probe \
//          -framework Accelerate
// run:   ./bench_c_graph_probe [N]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <arm_neon.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#define B 512
#define K 16

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    long N = (argc > 1) ? atol(argv[1]) : 100000;

    bnns_graph_compile_options_t opts = BNNSGraphCompileOptionsMakeDefault();
    bnns_graph_t graph = BNNSGraphCompileFromFile(
        "assets/coreml/matmul_512.mlmodelc", NULL, opts);
    BNNSGraphCompileOptionsDestroy(opts);
    if (!graph.data) { fprintf(stderr, "compile failed\n"); return 1; }

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
    fprintf(stderr, "graph: %zu inputs, %zu outputs, ws=%zu bytes\n", n_in, n_out, ws_size);

    // fp16 buffers: input X (B*K), output Y (B*K).
    uint16_t *X = aligned_alloc(64, sizeof(uint16_t) * B * K);
    uint16_t *Y = aligned_alloc(64, sizeof(uint16_t) * B * K);
    for (int i = 0; i < B * K; i++) X[i] = 0x3c00; // fp16 1.0

    // Outputs come first in arguments[].
    bnns_graph_argument_t args[2] = {0};
    args[0].data_ptr = Y; args[0].data_ptr_size = sizeof(uint16_t) * B * K;
    args[1].data_ptr = X; args[1].data_ptr_size = sizeof(uint16_t) * B * K;

    // Warm up.
    for (int i = 0; i < 1000; i++) {
        if (BNNSGraphContextExecute(ctx, NULL, 2, args, ws_size, ws) != 0) {
            fprintf(stderr, "execute failed\n"); return 1;
        }
    }

    double t0 = now_sec();
    for (long i = 0; i < N; i++) {
        BNNSGraphContextExecute(ctx, NULL, 2, args, ws_size, ws);
    }
    double t1 = now_sec();
    double per_call_us = (t1 - t0) / N * 1e6;
    double calls_per_sec = N / (t1 - t0);
    printf("BNNSGraph matmul B=%d: %.2f us/call, %.0f calls/sec\n",
           B, per_call_us, calls_per_sec);

    free(X); free(Y); free(ws);
    return 0;
}
