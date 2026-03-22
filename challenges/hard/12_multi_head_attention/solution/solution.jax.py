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


def _torch_impl(Q: Any, K: Any, V: Any, N: int, d_model: int, h: int):
    d_k = d_model // h
    scale = math.sqrt(d_k)
    pieces = []
    for head in range(h):
        start = head * d_k
        stop = start + d_k
        q_h = Q[:, start:stop]
        k_h = K[:, start:stop]
        v_h = V[:, start:stop]
        scores = torch.matmul(q_h, k_h.t()) / scale
        weights = torch.softmax(scores, dim=-1)
        pieces.append(torch.matmul(weights, v_h))
    return torch.cat(pieces, dim=-1)


def _jax_impl(Q: Any, K: Any, V: Any, N: int, d_model: int, h: int):
    d_k = d_model // h
    scale = math.sqrt(d_k)
    pieces = []
    for head in range(h):
        start = head * d_k
        stop = start + d_k
        q_h = Q[:, start:stop]
        k_h = K[:, start:stop]
        v_h = V[:, start:stop]
        scores = jnp.matmul(q_h, k_h.T) / scale
        scores = scores - jnp.max(scores, axis=-1, keepdims=True)
        weights = jnp.exp(scores)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        pieces.append(jnp.matmul(weights, v_h))
    return jnp.concatenate(pieces, axis=-1)


if jax is None:

    def solve(Q: Any, K: Any, V: Any, N: int, d_model: int, h: int):
        return _torch_impl(Q, K, V, N, d_model, h)

else:
    solve = jax.jit(_jax_impl, static_argnames=("N", "d_model", "h"))
