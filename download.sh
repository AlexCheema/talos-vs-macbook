#!/usr/bin/env bash
# Fetch microGPT weights from the upstream TALOS-V2 repo.
# We don't vendor them here because TALOS-V2 has no license — go check with
# the original authors before redistributing.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p assets
base="https://raw.githubusercontent.com/Luthiraa/TALOS-V2/main/rtl/microgpt"

for f in weights_only.npy microgpt.py names.txt; do
  if [[ ! -f "assets/$f" ]]; then
    echo "fetching $f..."
    curl -fsSL -o "assets/$f" "$base/$f"
  fi
done
echo "ok"
