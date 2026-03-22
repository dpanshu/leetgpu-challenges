from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

import torch


def _solve_impl(data, N: int):
    if jax is None:
        return torch.sort(data)[0]
    return jnp.sort(data)


if jax is None:

    def solve(data, N: int):
        return _solve_impl(data, N)

else:
    solve = jax.jit(_solve_impl)
