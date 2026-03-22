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


def _torch_impl(Q: Any, K: Any, V: Any, M: int, d: int):
    scores = torch.matmul(Q, K.t()) / math.sqrt(d)
    mask = torch.arange(M, device=Q.device)
    mask = mask[None, :] > mask[:, None]
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


def _jax_impl(Q: Any, K: Any, V: Any, M: int, d: int):
    scores = jnp.matmul(Q, K.T) / math.sqrt(d)
    idxs = jnp.arange(M)
    mask = idxs[None, :] > idxs[:, None]
    scores = jnp.where(mask, -jnp.inf, scores)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.matmul(weights, V)


if jax is None:

    def solve(Q: Any, K: Any, V: Any, M: int, d: int):
        return _torch_impl(Q, K, V, M, d)

else:
    solve = jax.jit(_jax_impl, static_argnames=("M", "d"))
