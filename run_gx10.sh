#!/usr/bin/env bash
# One-shot for NVIDIA DGX Spark (GX10): Grace ARM CPU + Blackwell GPU.
# Same shape as ./run.sh; emits the same line format so the awk parser is shared.
set -euo pipefail
cd "$(dirname "$0")"

# CUDA toolkit ships off-PATH on stock DGX Spark.
export PATH="/usr/local/cuda/bin:$PATH"

./download.sh
python3 convert_weights.py >/dev/null
make -s

TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT

{
  python3 pure_python.py --n 2000     --warmup 200
  python3 bench_numpy.py  --n 200000  --warmup 20000
  ./bench_c               2000000 100000
  ./bench_c_q412          2000000 100000
  ./bench_cuda            1000000  50000
  ./bench_cuda_persistent 2000000 100000
} | tee "$TMP" >/dev/null

echo
echo "=== microGPT inference on GX10 (Grace ARM + Blackwell), batch=1, char-by-char ==="
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
