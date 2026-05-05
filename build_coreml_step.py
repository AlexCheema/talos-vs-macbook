"""Build a CoreML mlpackage for one full microGPT inference step,
batched over B independent streams. Weights baked in (fp16). Inputs:
  tok    int32  [B]
  pos    int32  [B]
  K_in   fp16   [B, BLOCK, EMBD]
  V_in   fp16   [B, BLOCK, EMBD]
  u      fp16   [B]            (uniform random for inverse-CDF sample)
Outputs:
  tok_new int32 [B]
  pos_new int32 [B]
  K_out   fp16  [B, BLOCK, EMBD]
  V_out   fp16  [B, BLOCK, EMBD]

usage: build_coreml_step.py BATCH_SIZE OUTPUT_PACKAGE_PATH
"""

import sys
import math
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb

B = int(sys.argv[1])
OUT = sys.argv[2]

VOCAB = 27
BLOCK = 16
N_HEAD = 4
N_EMBD = 16
HEAD_DIM = N_EMBD // N_HEAD
MLP_HIDDEN = 4 * N_EMBD
BOS = 26
TEMPERATURE = 0.5

W = np.load("assets/weights_only.npy", allow_pickle=True).item()


def f16(arr):
    return arr.astype(np.float16)


WTE = f16(W["wte"])
WPE = f16(W["wpe"])
WQ  = f16(W["layer0.attn_wq"])
WK  = f16(W["layer0.attn_wk"])
WV  = f16(W["layer0.attn_wv"])
WO  = f16(W["layer0.attn_wo"])
W1  = f16(W["layer0.mlp_fc1"])
W2  = f16(W["layer0.mlp_fc2"])
LM  = f16(W["lm_head"])

INV_SQRT_HD = np.float16(1.0 / math.sqrt(HEAD_DIM))
INV_TEMP    = np.float16(1.0 / TEMPERATURE)
NEG_LARGE   = np.float16(-65500.0)


@mb.program(
    input_specs=[
        mb.TensorSpec(shape=(B,), dtype=ct.converters.mil.mil.types.int32),
        mb.TensorSpec(shape=(B,), dtype=ct.converters.mil.mil.types.int32),
        mb.TensorSpec(shape=(B, BLOCK, N_EMBD), dtype=ct.converters.mil.mil.types.fp16),
        mb.TensorSpec(shape=(B, BLOCK, N_EMBD), dtype=ct.converters.mil.mil.types.fp16),
        mb.TensorSpec(shape=(B,), dtype=ct.converters.mil.mil.types.fp16),
    ],
    opset_version=ct.target.macOS15,
)
def step(tok, pos, K_in, V_in, u):
    wte = mb.const(val=WTE)
    wpe = mb.const(val=WPE)
    wq  = mb.const(val=WQ)
    wk  = mb.const(val=WK)
    wv  = mb.const(val=WV)
    wo  = mb.const(val=WO)
    w1  = mb.const(val=W1)
    w2  = mb.const(val=W2)
    lm  = mb.const(val=LM)

    # Embed: x = WTE[tok] + WPE[pos] -> (B, EMBD)
    e_tok = mb.gather(x=wte, indices=tok, axis=0)
    e_pos = mb.gather(x=wpe, indices=pos, axis=0)
    x = mb.add(x=e_tok, y=e_pos)
    x = mb.cast(x=x, dtype="fp16")

    # First rmsnorm.
    x_sq = mb.mul(x=x, y=x)
    mean = mb.reduce_mean(x=x_sq, axes=[-1], keep_dims=True)
    eps = mb.const(val=np.float16(1e-5))
    inv_rms = mb.rsqrt(x=mb.add(x=mean, y=eps))
    x = mb.mul(x=x, y=inv_rms)

    # Save residual; second rmsnorm.
    xr = x
    x_sq2 = mb.mul(x=x, y=x)
    mean2 = mb.reduce_mean(x=x_sq2, axes=[-1], keep_dims=True)
    inv_rms2 = mb.rsqrt(x=mb.add(x=mean2, y=eps))
    x = mb.mul(x=x, y=inv_rms2)

    # Q, K, V projections: (B, EMBD) @ W.T = (B, EMBD).
    q = mb.matmul(x=x, y=wq, transpose_y=True)
    k = mb.matmul(x=x, y=wk, transpose_y=True)
    v = mb.matmul(x=x, y=wv, transpose_y=True)

    # KV cache update: shape-stable scatter via one-hot mask.
    positions = mb.const(val=np.arange(BLOCK, dtype=np.int32))
    # one_hot[B, BLOCK] = (positions == pos[:, None])
    pos_b1 = mb.expand_dims(x=pos, axes=[1])               # (B, 1)
    pos_1t = mb.expand_dims(x=positions, axes=[0])         # (1, BLOCK)
    eq = mb.equal(x=pos_1t, y=pos_b1)                      # (B, BLOCK) bool
    one_hot = mb.cast(x=eq, dtype="fp16")
    one_hot = mb.expand_dims(x=one_hot, axes=[2])          # (B, BLOCK, 1)
    one = mb.const(val=np.float16(1.0))
    inv_one_hot = mb.sub(x=one, y=one_hot)
    k_b = mb.expand_dims(x=k, axes=[1])                    # (B, 1, EMBD)
    v_b = mb.expand_dims(x=v, axes=[1])
    K_out = mb.add(x=mb.mul(x=K_in, y=inv_one_hot),
                   y=mb.mul(x=one_hot, y=k_b))
    V_out = mb.add(x=mb.mul(x=V_in, y=inv_one_hot),
                   y=mb.mul(x=one_hot, y=v_b))

    # Attention with mask. SDPA wants shapes (B, N_HEAD, T, HEAD_DIM).
    q_h = mb.reshape(x=q, shape=(B, 1, N_HEAD, HEAD_DIM))
    q_h = mb.transpose(x=q_h, perm=[0, 2, 1, 3])           # (B, H, 1, D)
    k_h = mb.reshape(x=K_out, shape=(B, BLOCK, N_HEAD, HEAD_DIM))
    k_h = mb.transpose(x=k_h, perm=[0, 2, 1, 3])           # (B, H, T, D)
    v_h = mb.reshape(x=V_out, shape=(B, BLOCK, N_HEAD, HEAD_DIM))
    v_h = mb.transpose(x=v_h, perm=[0, 2, 1, 3])

    # Mask: True where positions <= pos (attendable). (B, 1, 1, BLOCK)
    le = mb.less_equal(x=pos_1t, y=pos_b1)                 # (B, BLOCK)
    mask_f = mb.cast(x=le, dtype="fp16")
    mask_f = mb.reshape(x=mask_f, shape=(B, 1, 1, BLOCK))
    add_mask = mb.mul(x=mb.sub(x=one, y=mask_f),
                      y=mb.const(val=NEG_LARGE))           # (B,1,1,BLOCK)

    # Manual SDPA so the additive mask is honored.
    scale = mb.const(val=INV_SQRT_HD)
    logits = mb.matmul(x=q_h, y=k_h, transpose_y=True)     # (B, H, 1, T)
    logits = mb.mul(x=logits, y=scale)
    logits = mb.add(x=logits, y=add_mask)
    aw = mb.softmax(x=logits, axis=-1)
    head_out = mb.matmul(x=aw, y=v_h)                      # (B, H, 1, D)
    head_out = mb.transpose(x=head_out, perm=[0, 2, 1, 3]) # (B, 1, H, D)
    head_out = mb.reshape(x=head_out, shape=(B, N_EMBD))

    # WO + residual.
    x = mb.matmul(x=head_out, y=wo, transpose_y=True)
    x = mb.add(x=x, y=xr)

    # MLP block: rmsnorm, fc1, relu, fc2, residual.
    xr2 = x
    x_sq3 = mb.mul(x=x, y=x)
    mean3 = mb.reduce_mean(x=x_sq3, axes=[-1], keep_dims=True)
    inv_rms3 = mb.rsqrt(x=mb.add(x=mean3, y=eps))
    x = mb.mul(x=x, y=inv_rms3)
    h = mb.matmul(x=x, y=w1, transpose_y=True)             # (B, MLP_HIDDEN)
    h = mb.relu(x=h)
    x = mb.matmul(x=h, y=w2, transpose_y=True)             # (B, EMBD)
    x = mb.add(x=x, y=xr2)

    # LM head + temperature softmax.
    logits_out = mb.matmul(x=x, y=lm, transpose_y=True)    # (B, VOCAB)
    inv_temp_c = mb.const(val=INV_TEMP)
    logits_out = mb.mul(x=logits_out, y=inv_temp_c)
    probs = mb.softmax(x=logits_out, axis=-1)

    # Inverse-CDF sample: argmax over (cumsum(probs) > u). Fall back to
    # VOCAB - 1 on rows where FP roundoff leaves the final cdf entry below
    # 1.0; reduce_argmax would otherwise return 0 on an all-False row.
    cdf = mb.cumsum(x=probs, axis=-1)
    u_b1 = mb.expand_dims(x=u, axes=[1])                   # (B, 1)
    gt = mb.greater(x=cdf, y=u_b1)                         # (B, VOCAB) bool
    gt_i32 = mb.cast(x=gt, dtype="int32")
    nxt = mb.reduce_argmax(x=gt_i32, axis=-1, keep_dims=False)
    nxt = mb.cast(x=nxt, dtype="int32")
    any_above = mb.cast(x=mb.reduce_max(x=gt_i32, axes=[-1], keep_dims=False),
                        dtype="bool")
    last_tok = mb.const(val=np.int32(VOCAB - 1))
    nxt = mb.select(cond=any_above, a=nxt, b=last_tok)

    # Reset/advance: if nxt == BOS, pos_new=0, tok_new=BOS;
    #                else if pos+1 >= BLOCK, pos_new=0, tok_new=BOS;
    #                else pos_new = pos+1, tok_new = nxt.
    bos_c    = mb.const(val=np.int32(BOS))
    zero_c   = mb.const(val=np.int32(0))
    one_i32  = mb.const(val=np.int32(1))
    block_c  = mb.const(val=np.int32(BLOCK))
    pos_p1   = mb.add(x=pos, y=one_i32)
    overflow = mb.greater_equal(x=pos_p1, y=block_c)
    is_bos   = mb.equal(x=nxt, y=bos_c)
    reset    = mb.logical_or(x=is_bos, y=overflow)
    pos_new  = mb.select(cond=reset, a=zero_c, b=pos_p1)
    tok_new  = mb.select(cond=reset, a=bos_c, b=nxt)

    return tok_new, pos_new, K_out, V_out


model = ct.convert(
    step,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
)
model.save(OUT)
print(f"saved {OUT}")
