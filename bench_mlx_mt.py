"""MLX multi-threaded batched throughput.

Splits B streams across T Python threads, each driving its own MLX
async pipeline (cpu or gpu). MLX releases the Python GIL inside C
extension calls, but the Python-side per-step overhead (mx.array
allocs, np.fromiter for u, control flow) still serializes through
the GIL like numpy.

usage: bench_mlx_mt.py [--gpu] --batch B --workers T [--n N] [--warmup W]
"""

import os, sys, time, argparse, threading, random
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--warmup", type=int, default=200)
args = ap.parse_args()

import mlx.core as mx
device = mx.gpu if args.gpu else mx.cpu

# Reuse the batched build/step from bench_mlx.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_mlx import build_batched
from model import VOCAB_SIZE, BLOCK_SIZE, N_EMBD, BOS

if args.batch % args.workers:
    sys.exit("--batch must be multiple of --workers")
B_local = args.batch // args.workers


class Worker(threading.Thread):
    def __init__(self, tid, start_bar, end_bar):
        super().__init__()
        self.tid = tid
        self.start_bar = start_bar
        self.end_bar = end_bar

    def run(self):
        mx.set_default_device(device)
        forward_b, step_graph_b = build_batched(device, B_local)
        rng = random.Random(42 + self.tid * B_local)
        state = {
            "tok": mx.full((B_local,), BOS, dtype=mx.int32),
            "pos": mx.zeros((B_local,), dtype=mx.int32),
            "K": mx.zeros((B_local, BLOCK_SIZE, N_EMBD), dtype=mx.float32),
            "V": mx.zeros((B_local, BLOCK_SIZE, N_EMBD), dtype=mx.float32),
        }

        def step():
            u = np.fromiter((rng.random() for _ in range(B_local)),
                            dtype=np.float32, count=B_local)
            u_arr = mx.array(u)
            nt, np_, K_new, V_new = step_graph_b(state["tok"], state["pos"],
                                                  u_arr, state["K"], state["V"])
            state["tok"], state["pos"], state["K"], state["V"] = nt, np_, K_new, V_new

        def flush():
            mx.eval(state["tok"], state["pos"], state["K"], state["V"])

        # warmup
        for i in range(args.warmup):
            step()
            if (i + 1) % 64 == 0:
                flush()
        flush()

        self.start_bar.wait()
        for i in range(args.n):
            step()
            if (i + 1) % 64 == 0:
                flush()
        flush()
        self.end_bar.wait()


def main():
    start_bar = threading.Barrier(args.workers + 1)
    end_bar = threading.Barrier(args.workers + 1)
    workers = [Worker(t, start_bar, end_bar) for t in range(args.workers)]
    for w in workers: w.start()
    start_bar.wait()
    t0 = time.perf_counter()
    end_bar.wait()
    t1 = time.perf_counter()
    for w in workers: w.join()

    rate = args.batch * args.n / (t1 - t0)
    dev = "gpu" if args.gpu else "cpu"
    label = f"mlx fp32 ({dev} batch={args.batch} t={args.workers})"
    print(f"  {label:24s}  {rate:>14,.0f} tok/sec")


main()
