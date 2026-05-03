#!/usr/bin/env bash
# One-shot: fetch weights, build, benchmark.
set -euo pipefail
cd "$(dirname "$0")"

./download.sh
python3 convert_weights.py >/dev/null
make -s

TMP_SINGLE=$(mktemp); TMP_BATCH=$(mktemp); TMP_MT=$(mktemp)
trap 'rm -f "$TMP_SINGLE" "$TMP_BATCH" "$TMP_MT"' EXIT

# Single-stream benchmarks (batch=1, char-by-char, FPGA-comparable).
{
  python3 pure_python.py --n 2000     --warmup 200
  python3 bench_numpy.py  --n 200000  --warmup 20000
  if python3 -c "import mlx.core" 2>/dev/null; then
    python3 bench_mlx.py    --n 50000  --warmup 2000
    python3 bench_mlx.py --gpu --n 20000 --warmup 1000
    python3 bench_mlx.py --async --n 50000 --warmup 2000
    python3 bench_mlx.py --gpu --async --n 50000 --warmup 2000
    python3 bench_mlx.py --gpu --async --rollout 8 --n 50000 --warmup 2000
  else
    echo "  mlx not installed                       skipped (pip install mlx)"
  fi
  ./bench_c       2000000 100000
  ./bench_c_q412  2000000 100000
} | tee "$TMP_SINGLE" >/dev/null

# Batched throughput benchmarks (different problem: N independent streams).
# Skipped for pure-python (no per-call overhead to amortize).
{
  python3 bench_numpy.py --batch 8   --n 10000 --warmup 1000
  python3 bench_numpy.py --batch 64  --n 5000  --warmup 500
  python3 bench_numpy.py --batch 512 --n 2000  --warmup 200
  if python3 -c "import mlx.core" 2>/dev/null; then
    python3 bench_mlx.py --batch 8   --n 10000 --warmup 1000
    python3 bench_mlx.py --batch 64  --n 5000  --warmup 500
    python3 bench_mlx.py --gpu --batch 8   --n 10000 --warmup 1000
    python3 bench_mlx.py --gpu --batch 64  --n 5000  --warmup 500
    python3 bench_mlx.py --gpu --batch 512 --n 2000  --warmup 200
    python3 bench_mlx.py --gpu --batch 512 --rollout 4 --n 1000 --warmup 100
  fi
  ./bench_c_batch 8   2000000 100000
  ./bench_c_batch 32  500000  25000
  ./bench_c_batch 128 100000  5000
  ./bench_c_sme   8   2000000 100000
  ./bench_c_sme   128 100000  5000
  ./bench_c_sme   1024 10000  500
} | tee "$TMP_BATCH" >/dev/null

# Multi-threaded throughput: scale across CPU cores. M5 Max = 12 P + 6 E.
{
  python3 bench_numpy_mt.py --batch 12096 --workers 12 --n 200 --warmup 30
  if python3 -c "import mlx.core" 2>/dev/null; then
    python3 bench_mlx_mt.py --batch 12096 --workers 12 --n 200 --warmup 30
  fi
  ./bench_c_batch_mt 384  12 1000000 50000   # NEON, 12 P-cores
  ./bench_c_batch_mt 576  18 1000000 50000   # NEON, all 18 cores
  ./bench_c_sme_mt   3072 12 200000  10000   # SME2, 12 P-cores, B/thr=256
  ./bench_c_sme_mt   4608 18 200000  10000   # SME2, all 18 cores, B/thr=256
} | tee "$TMP_MT" >/dev/null

print_table() {
  awk '
    /skipped/ { print; next }
    /^[[:space:]]/ {
      label = $1
      for (i = 2; i <= NF - 2; i++) label = label " " $i
      rate = $(NF - 1); gsub(",", "", rate); rate += 0
      printf "  %-28s %14d %11.2fx\n", label, rate, rate / 53000.0
    }
  ' "$1"
}

echo
echo "=== microGPT inference, single-thread, batch=1, char-by-char ==="
echo
printf "  %-28s %14s %12s\n" "implementation" "tok/sec" "vs FPGA"
printf "  %-28s %14s %12s\n" "----------------------------" "--------------" "------------"
print_table "$TMP_SINGLE"
printf "  %-28s %14d %11s\n" "TALOS-V2 (FPGA, 56MHz)" 53000 "1.00x"

if [ -s "$TMP_BATCH" ]; then
  echo
  echo "=== batched throughput (N independent streams, total tok/sec) ==="
  echo
  printf "  %-28s %14s %12s\n" "implementation" "tok/sec" "vs FPGA"
  printf "  %-28s %14s %12s\n" "----------------------------" "--------------" "------------"
  print_table "$TMP_BATCH"
fi

if [ -s "$TMP_MT" ]; then
  echo
  echo "=== multi-threaded throughput (CPU cores in parallel, total tok/sec) ==="
  echo
  printf "  %-28s %14s %12s\n" "implementation" "tok/sec" "vs FPGA"
  printf "  %-28s %14s %12s\n" "----------------------------" "--------------" "------------"
  print_table "$TMP_MT"
fi
echo
