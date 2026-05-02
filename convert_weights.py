"""Flatten weights_only.npy into raw binary blobs the C benchmarks can mmap.

Produces:
  assets/weights_fp32.bin   - 4192 float32, in WEIGHT_ORDER from model.py
  assets/weights_q412.bin   - 4192 int16  (round(w * 4096), clamped to [-32768, 32767])
"""

import numpy as np
from model import load_weights, WEIGHT_ORDER, ASSETS
import os

w = load_weights()

flat = np.concatenate([w[k].reshape(-1) for k in WEIGHT_ORDER]).astype(np.float32)
print(f"total params: {flat.size}")
flat.tofile(os.path.join(ASSETS, "weights_fp32.bin"))

q = np.clip(np.round(flat * 4096.0), -32768, 32767).astype(np.int16)
q.tofile(os.path.join(ASSETS, "weights_q412.bin"))

# Quick sanity: max abs error from quantization
recon = q.astype(np.float32) / 4096.0
err = np.max(np.abs(recon - flat))
print(f"q4.12 max abs reconstruction error: {err:.6f}")
print("ok")
