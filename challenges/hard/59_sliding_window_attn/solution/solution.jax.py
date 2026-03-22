from __future__ import annotations

from typing import Any
import math

import torch

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


def _torch_impl(Q: Any, K: Any, V: Any, M: int, d: int, window_size: int):
    scores = torch.matmul(Q, K.t()) / math.sqrt(d)
    idxs = torch.arange(M, device=Q.device)
    mask = (idxs[None, :] - idxs[:, None]).abs() > window_size
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V


def _jax_impl(Q: Any, K: Any, V: Any, M: int, d: int, window_size: int):
    scores = jnp.matmul(Q, K.T) / math.sqrt(d)
    idxs = jnp.arange(M)
    mask = jnp.abs(idxs[None, :] - idxs[:, None]) > window_size
    scores = jnp.where(mask, -jnp.inf, scores)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    weights = jnp.exp(scores)
    weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
    return jnp.matmul(weights, V)


if jax is None:

    def solve(Q: Any, K: Any, V: Any, M: int, d: int, window_size: int):
        return _torch_impl(Q, K, V, M, d, window_size)

else:
    solve = jax.jit(_jax_impl, static_argnames=("M", "d", "window_size"))
