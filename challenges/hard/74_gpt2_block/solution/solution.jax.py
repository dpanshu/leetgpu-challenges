from __future__ import annotations

from typing import Any
import math

import torch
import torch.nn.functional as F

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


D = 768
H = 12
DH = D // H
FFN = 3072

O_LN1_W = 0
O_LN1_B = O_LN1_W + D
O_WQKV = O_LN1_B + D
O_BQKV = O_WQKV + D * 3 * D
O_WAPROJ = O_BQKV + 3 * D
O_BAPROJ = O_WAPROJ + D * D
O_LN2_W = O_BAPROJ + D
O_LN2_B = O_LN2_W + D
O_WFC = O_LN2_B + D
O_BFC = O_WFC + D * FFN
O_WPROJ = O_BFC + FFN
O_BPROJ = O_WPROJ + FFN * D
TOTAL_WEIGHTS = O_BPROJ + D


def _torch_layer_norm(x, weight, bias, eps=1e-5):
    return F.layer_norm(x, [D], weight, bias, eps=eps)


def _torch_impl(x: Any, output: Any, weights: Any, seq_len: int):
    ln1_w = weights[O_LN1_W:O_LN1_B]
    ln1_b = weights[O_LN1_B:O_WQKV]
    W_qkv = weights[O_WQKV:O_BQKV].view(D, 3 * D)
    b_qkv = weights[O_BQKV:O_WAPROJ]
    W_attn = weights[O_WAPROJ:O_BAPROJ].view(D, D)
    b_attn = weights[O_BAPROJ:O_LN2_W]
    ln2_w = weights[O_LN2_W:O_LN2_B]
    ln2_b = weights[O_LN2_B:O_WFC]
    W_fc = weights[O_WFC:O_BFC].view(D, FFN)
    b_fc = weights[O_BFC:O_WPROJ]
    W_proj = weights[O_WPROJ:O_BPROJ].view(FFN, D)
    b_proj = weights[O_BPROJ : O_BPROJ + D]

    x_norm = _torch_layer_norm(x, ln1_w, ln1_b)
    qkv = x_norm @ W_qkv + b_qkv
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(seq_len, H, DH).transpose(0, 1)
    k = k.view(seq_len, H, DH).transpose(0, 1)
    v = v.view(seq_len, H, DH).transpose(0, 1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(DH)
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)
    attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, D)
    attn_proj = attn_out @ W_attn + b_attn
    hidden = x + attn_proj

    h_norm = _torch_layer_norm(hidden, ln2_w, ln2_b)
    fc = F.gelu(h_norm @ W_fc + b_fc, approximate="tanh")
    proj = fc @ W_proj + b_proj
    output.copy_(hidden + proj)


def _gelu_tanh(x):
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _layer_norm(x, weight, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    x_hat = (x - mean) / jnp.sqrt(var + eps)
    return x_hat * weight + bias


def _jax_impl(x: Any, weights: Any, seq_len: int):
    ln1_w = weights[O_LN1_W:O_LN1_B]
    ln1_b = weights[O_LN1_B:O_WQKV]
    W_qkv = weights[O_WQKV:O_BQKV].reshape(D, 3 * D)
    b_qkv = weights[O_BQKV:O_WAPROJ]
    W_attn = weights[O_WAPROJ:O_BAPROJ].reshape(D, D)
    b_attn = weights[O_BAPROJ:O_LN2_W]
    ln2_w = weights[O_LN2_W:O_LN2_B]
    ln2_b = weights[O_LN2_B:O_WFC]
    W_fc = weights[O_WFC:O_BFC].reshape(D, FFN)
    b_fc = weights[O_BFC:O_WPROJ]
    W_proj = weights[O_WPROJ:O_BPROJ].reshape(FFN, D)
    b_proj = weights[O_BPROJ : O_BPROJ + D]

    x_norm = _layer_norm(x, ln1_w, ln1_b)
    qkv = x_norm @ W_qkv + b_qkv
    q = qkv[:, :D]
    k = qkv[:, D : 2 * D]
    v = qkv[:, 2 * D :]
    q = q.reshape(seq_len, H, DH).transpose(1, 0, 2)
    k = k.reshape(seq_len, H, DH).transpose(1, 0, 2)
    v = v.reshape(seq_len, H, DH).transpose(1, 0, 2)

    scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / math.sqrt(DH)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights_attn = jnp.exp(scores)
    weights_attn = weights_attn / jnp.sum(weights_attn, axis=-1, keepdims=True)
    attn_out = jnp.matmul(weights_attn, v)
    attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, D)
    attn_proj = attn_out @ W_attn + b_attn
    hidden = x + attn_proj

    h_norm = _layer_norm(hidden, ln2_w, ln2_b)
    fc = _gelu_tanh(h_norm @ W_fc + b_fc)
    proj = fc @ W_proj + b_proj
    return hidden + proj


if jax is None:

    def solve(x: Any, output: Any, weights: Any, seq_len: int):
        _torch_impl(x, output, weights, seq_len)

else:
    solve = jax.jit(_jax_impl, static_argnames=("seq_len",))
