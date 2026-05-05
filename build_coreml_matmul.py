"""Build a CoreML mlpackage that does a single bf16 matmul Y = X @ W.T,
weights baked in. Used to probe BNNSGraph per-call overhead vs BNNSMatMul.

usage: build_coreml_matmul.py BATCH_SIZE OUTPUT_DIR
  produces matmul_BATCH.mlpackage (then compile with:
    xcrun coremlcompiler compile <pkg> <out>)
"""

import sys
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

B = int(sys.argv[1])
OUT = sys.argv[2]
EMBD = 16

# Random fp32 weights, will be baked into the program as bf16.
np.random.seed(0)
W = np.random.randn(EMBD, EMBD).astype(np.float32) * 0.1


@mb.program(
    input_specs=[mb.TensorSpec(shape=(B, EMBD), dtype=types.fp16)],
    opset_version=ct.target.macOS15,
)
def prog(x):
    w = mb.const(val=W.astype(np.float16))
    # Y = X @ W.T  i.e. matmul(x, w, transpose_y=True)
    return mb.matmul(x=x, y=w, transpose_y=True)


model = ct.convert(
    prog,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
)

model.save(OUT)
print(f"saved {OUT}")
