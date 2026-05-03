#!/usr/bin/env bash
# One-shot: fetch weights, build, benchmark.
set -euo pipefail
cd "$(dirname "$0")"

./download.sh
python3 convert_weights.py >/dev/null
make -s

TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

{
  python3 pure_python.py --n 2000     --warmup 200
  python3 bench_numpy.py  --n 200000  --warmup 20000
  if python3 -c "import mlx.core" 2>/dev/null; then
    python3 bench_mlx.py    --n 50000  --warmup 2000
    python3 bench_mlx.py --gpu --n 20000 --warmup 1000
    python3 bench_mlx.py --async --n 50000 --warmup 2000
    python3 bench_mlx.py --gpu --async --n 50000 --warmup 2000
  else
    echo "  mlx not installed                       skipped (pip install mlx)"
  fi
  ./bench_c       2000000 100000
  ./bench_c_q412  2000000 100000
} | tee "$TMP" >/dev/null

echo
echo "=== microGPT inference, single-thread, batch=1, char-by-char ==="
echo
printf "  %-28s %14s %12s\n" "implementation" "tok/sec" "vs FPGA"
printf "  %-28s %14s %12s\n" "----------------------------" "--------------" "------------"

awk '
  /skipped/ { print; next }
  /^[[:space:]]/ {
    label = $1
    for (i = 2; i <= NF - 2; i++) label = label " " $i
    rate = $(NF - 1); gsub(",", "", rate); rate += 0
    printf "  %-28s %14d %11.2fx\n", label, rate, rate / 53000.0
  }
' "$TMP"

printf "  %-28s %14d %11s\n" "TALOS-V2 (FPGA, 56MHz)" 53000 "1.00x"
echo
