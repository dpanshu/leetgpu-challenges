from __future__ import annotations

from typing import Any

import torch

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


def _torch_impl(Q: Any, K: Any, V: Any, M: int, d: int):
    phi_Q = torch.where(Q > 0, Q + 1, torch.exp(Q))
    phi_K = torch.where(K > 0, K + 1, torch.exp(K))
    S = phi_K.t() @ V
    z = phi_K.sum(dim=0)
    numerator = phi_Q @ S
    denominator = phi_Q @ z
    return numerator / denominator.unsqueeze(-1)


def _jax_impl(Q: Any, K: Any, V: Any, M: int, d: int):
    phi_Q = jnp.where(Q > 0, Q + 1, jnp.exp(Q))
    phi_K = jnp.where(K > 0, K + 1, jnp.exp(K))
    S = phi_K.T @ V
    z = phi_K.sum(axis=0)
    numerator = phi_Q @ S
    denominator = phi_Q @ z
    return numerator / denominator[:, None]


if jax is None:

    def solve(Q: Any, K: Any, V: Any, M: int, d: int):
        return _torch_impl(Q, K, V, M, d)

else:
    solve = jax.jit(_jax_impl, static_argnames=("M", "d"))
