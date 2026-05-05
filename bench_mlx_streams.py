"""Probe: do MLX streams give parallel GPU execution on M-series Macs?

Compares N concurrent MLX streams (each with B/N batch) vs one stream
with the full B as a baseline. If the GPU can run streams in parallel,
N streams should approach 1-stream throughput. They don't.

usage: bench_mlx_streams.py [--batch B] [--streams N1,N2,...] [--n N]
"""
import sys, os, time, argparse, random, threading
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=4096,
                help="Total batch — split across streams")
ap.add_argument("--streams", type=str, default="1,2,4,8",
                help="Comma-sep list of stream counts to try")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--warmup", type=int, default=50)
args = ap.parse_args()

import mlx.core as mx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_mlx import build_batched
from model import BLOCK_SIZE, N_EMBD, BOS


def make_runner(stream, B):
    with mx.stream(stream):
        forward_b, step_graph_b = build_batched(mx.gpu, B)
    rng = random.Random(42 + B)
    state = {
        "tok": mx.full((B,), BOS, dtype=mx.int32),
        "pos": mx.zeros((B,), dtype=mx.int32),
        "K": mx.zeros((B, BLOCK_SIZE, N_EMBD), dtype=mx.float32),
        "V": mx.zeros((B, BLOCK_SIZE, N_EMBD), dtype=mx.float32),
    }
    def step():
        with mx.stream(stream):
            u = mx.array(np.fromiter((rng.random() for _ in range(B)),
                                     dtype=np.float32, count=B))
            nt, np_, K_new, V_new = step_graph_b(state["tok"], state["pos"], u,
                                                  state["K"], state["V"])
            state["tok"], state["pos"], state["K"], state["V"] = nt, np_, K_new, V_new
    def flush():
        mx.eval(state["tok"], state["pos"], state["K"], state["V"])
    return step, flush


for n in (int(s) for s in args.streams.split(",")):
    if args.batch % n: continue
    B_per = args.batch // n
    streams = [mx.new_stream(mx.gpu) for _ in range(n)]
    runners = [make_runner(s, B_per) for s in streams]

    for _ in range(args.warmup):
        for step, _ in runners: step()
    for _, flush in runners: flush()

    t0 = time.perf_counter()
    for i in range(args.n):
        for step, _ in runners: step()
        if (i + 1) % 64 == 0:
            for _, flush in runners: flush()
    for _, flush in runners: flush()
    t1 = time.perf_counter()

    rate = args.batch * args.n / (t1 - t0)
    label = f"mlx fp32 (gpu B={args.batch} streams={n})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")
