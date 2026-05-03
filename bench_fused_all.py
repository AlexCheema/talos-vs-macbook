"""Run the GPU Metal-kernel implementation AND the CPU SME2/NEON
batched implementation concurrently, summing total throughput.

The GPU runs `bench_mlx_metal`'s in-kernel autoregressive loop. The CPU
runs `bench_c_sme_mt` (or `bench_c_batch_mt`) as a child process so it
gets its own pthreads and isn't blocked by Python's GIL.

Each side reports its own tokens/sec; this driver waits the same wall
time for both and sums them.

usage: bench_fused_all.py [--cpu-impl sme2|neon] [--cpu-threads T]
                          [--cpu-batch-per B] [--gpu-streams S]
                          [--gpu-steps N] [--seconds T]
"""
import os, sys, time, argparse, threading, subprocess, signal
import numpy as np
import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--cpu-impl", choices=("sme2", "neon"), default="sme2")
ap.add_argument("--cpu-threads", type=int, default=12)
ap.add_argument("--cpu-batch-per", type=int, default=256,
                help="Batch per CPU thread; total CPU batch = T * B-per")
ap.add_argument("--gpu-streams", type=int, default=8192)
ap.add_argument("--gpu-steps", type=int, default=256)
ap.add_argument("--seconds", type=float, default=5.0,
                help="Run roughly this long, then report.")
ap.add_argument("--gpu-impl", default="tgkv",
                choices=("fp32", "fp16", "simd", "tgkv"),
                help="Which Metal kernel variant to use on the GPU.")
args = ap.parse_args()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_MOD = {
    "fp32": "bench_mlx_metal",
    "fp16": "bench_mlx_metal_fp16",
    "simd": "bench_mlx_metal_simd",
    "tgkv": "bench_mlx_metal_tgkv",
}[args.gpu_impl]
_GPU_HAS_KV_OUTPUT = args.gpu_impl != "tgkv"
import importlib
_gpu_mod = importlib.import_module(_MOD)
KERNEL, BLOCK, EMBD = _gpu_mod.KERNEL, _gpu_mod.BLOCK, _gpu_mod.EMBD
_GPU_DTYPE = mx.float32 if args.gpu_impl == "fp32" else mx.float16
_GPU_BENCH = _MOD + ".py"


def _shapes_dtypes(S, N, kv):
    if _GPU_HAS_KV_OUTPUT:
        return ([(S * N,), (S,), (kv,), (kv,)],
                [mx.uint32, mx.uint32, _GPU_DTYPE, _GPU_DTYPE])
    return ([(S * N,), (S,)], [mx.uint32, mx.uint32])


def gpu_worker(stop_evt, result):
    mx.set_default_device(mx.gpu)
    W_flat = _gpu_mod.load_weights()
    W = mx.array(W_flat)
    seeds = mx.array(np.arange(1, args.gpu_streams + 1, dtype=np.uint32))
    n_steps = mx.array(np.array([args.gpu_steps], dtype=np.uint32))
    kv = args.gpu_streams * BLOCK * EMBD
    shapes, dtypes = _shapes_dtypes(args.gpu_streams, args.gpu_steps, kv)

    for _ in range(3):
        outs = KERNEL(inputs=[W, seeds, n_steps], grid=(args.gpu_streams * 32, 1, 1),
                      threadgroup=(32, 1, 1), output_shapes=shapes, output_dtypes=dtypes)
        mx.eval(outs[0], outs[1])
        seeds = outs[1]

    total = 0
    t0 = time.perf_counter()
    while not stop_evt.is_set():
        outs = KERNEL(inputs=[W, seeds, n_steps], grid=(args.gpu_streams * 32, 1, 1),
                      threadgroup=(32, 1, 1), output_shapes=shapes, output_dtypes=dtypes)
        mx.eval(outs[0], outs[1])
        seeds = outs[1]
        total += args.gpu_streams * args.gpu_steps
    t1 = time.perf_counter()
    result["tokens"] = total
    result["secs"] = t1 - t0


def cpu_worker(stop_evt, result):
    # Use a long, large benchmark so it runs until we kill it.
    bin_name = "bench_c_sme_mt" if args.cpu_impl == "sme2" else "bench_c_batch_mt"
    binary = os.path.join(HERE, bin_name)
    total_b = args.cpu_threads * args.cpu_batch_per
    # Big N so it doesn't finish before --seconds; we'll send SIGTERM.
    cmd = [binary, str(total_b), str(args.cpu_threads), "100000000", "100"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            preexec_fn=os.setsid)
    # Give it a moment to warm up, then mark t0 (we measure FROM the moment
    # GPU starts — but the CPU process counts from its own warmup-end internally).
    # Simplest: just kill it after --seconds and compute its own runtime.
    t0 = time.perf_counter()
    stop_evt.wait()
    t1 = time.perf_counter()
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait(timeout=2)
    # We can't easily get tokens/sec from the killed process (it never printed).
    # Fall back: run it for a fixed duration externally and use its own report.
    result["wall"] = t1 - t0


def measure_cpu_alone():
    """Run the CPU bench alone for a fixed N that takes ~args.seconds, return tok/s."""
    bin_name = "bench_c_sme_mt" if args.cpu_impl == "sme2" else "bench_c_batch_mt"
    binary = os.path.join(HERE, bin_name)
    total_b = args.cpu_threads * args.cpu_batch_per
    # Calibrate N: do a short run (~0.5s) to estimate steps/sec, then size N for ~args.seconds.
    short = subprocess.run([binary, str(total_b), str(args.cpu_threads), "200", "20"],
                           capture_output=True, text=True, check=True)
    parts = short.stdout.split()
    rate = float(parts[-2].replace(",", ""))  # tok/sec
    n_for_seconds = max(100, int(rate * args.seconds / total_b))
    out = subprocess.run([binary, str(total_b), str(args.cpu_threads), str(n_for_seconds), "100"],
                         capture_output=True, text=True, check=True)
    parts = out.stdout.split()
    return float(parts[-2].replace(",", "")), out.stdout.strip()


def measure_gpu_alone():
    """Run the GPU bench alone for ~args.seconds."""
    cmd = [sys.executable, os.path.join(HERE, _GPU_BENCH),
           "--streams", str(args.gpu_streams), "--steps", str(args.gpu_steps),
           "--reps", "5", "--warmup", "1"]
    short = subprocess.run(cmd, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONPATH": HERE})
    parts = short.stdout.strip().split()
    rate = float(parts[-2].replace(",", ""))
    reps_for_seconds = max(2, int(rate * args.seconds / (args.gpu_streams * args.gpu_steps)))
    cmd2 = [sys.executable, os.path.join(HERE, _GPU_BENCH),
            "--streams", str(args.gpu_streams), "--steps", str(args.gpu_steps),
            "--reps", str(reps_for_seconds), "--warmup", "2"]
    out = subprocess.run(cmd2, capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": HERE})
    parts = out.stdout.strip().split()
    return float(parts[-2].replace(",", "")), out.stdout.strip()


def main():
    print(f"=== fused: GPU metal kernel + CPU {args.cpu_impl} threaded ===")
    print(f"GPU: S={args.gpu_streams} N={args.gpu_steps}")
    print(f"CPU: impl={args.cpu_impl} threads={args.cpu_threads} "
          f"batch_per={args.cpu_batch_per} total_batch={args.cpu_threads * args.cpu_batch_per}")
    print()

    # Baseline 1: CPU alone for ~args.seconds.
    print(f"[baseline] CPU alone (~{args.seconds:.1f}s)...")
    cpu_alone_rate, cpu_alone_line = measure_cpu_alone()
    print(f"  -> {cpu_alone_line}")

    # Baseline 2: GPU alone for ~args.seconds.
    print(f"[baseline] GPU alone (~{args.seconds:.1f}s)...")
    gpu_alone_rate, gpu_alone_line = measure_gpu_alone()
    print(f"  -> {gpu_alone_line}")

    # Concurrent: spawn both, time ~args.seconds.
    print(f"[concurrent] GPU + CPU together (~{args.seconds:.1f}s)...")
    cpu_bin = "bench_c_sme_mt" if args.cpu_impl == "sme2" else "bench_c_batch_mt"
    cpu_binary = os.path.join(HERE, cpu_bin)
    cpu_total_b = args.cpu_threads * args.cpu_batch_per
    # Calibrate for cpu N to take ~args.seconds.
    cpu_n_estimate = max(100, int(cpu_alone_rate * args.seconds / cpu_total_b))
    cpu_proc = subprocess.Popen(
        [cpu_binary, str(cpu_total_b), str(args.cpu_threads), str(cpu_n_estimate), "100"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    gpu_stop = threading.Event()
    gpu_result = {}
    gpu_thr = threading.Thread(target=gpu_worker, args=(gpu_stop, gpu_result))
    gpu_thr.start()

    cpu_stdout, _ = cpu_proc.communicate()
    gpu_stop.set()
    gpu_thr.join()

    cpu_concurrent_line = cpu_stdout.strip()
    cpu_concurrent_rate = float(cpu_concurrent_line.split()[-2].replace(",", ""))
    gpu_concurrent_rate = gpu_result["tokens"] / gpu_result["secs"]

    print(f"  CPU: {cpu_concurrent_line}")
    print(f"  GPU: {gpu_concurrent_rate:>14,.0f} tok/sec  "
          f"({gpu_result['tokens']:,} tokens in {gpu_result['secs']:.2f}s)")
    print()
    total = cpu_concurrent_rate + gpu_concurrent_rate
    sum_alone = cpu_alone_rate + gpu_alone_rate
    print(f"=== Summary ===")
    print(f"  CPU alone:       {cpu_alone_rate:>14,.0f} tok/sec")
    print(f"  GPU alone:       {gpu_alone_rate:>14,.0f} tok/sec")
    print(f"  CPU concurrent:  {cpu_concurrent_rate:>14,.0f} tok/sec  "
          f"({100 * cpu_concurrent_rate / cpu_alone_rate:.0f}% of alone)")
    print(f"  GPU concurrent:  {gpu_concurrent_rate:>14,.0f} tok/sec  "
          f"({100 * gpu_concurrent_rate / gpu_alone_rate:.0f}% of alone)")
    print(f"  TOTAL fused:     {total:>14,.0f} tok/sec  "
          f"({100 * total / sum_alone:.0f}% of theoretical sum {sum_alone:,.0f})")


main()
