from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

import torch


def _solve_impl(input, N: int):
    if jax is None:
        return torch.sort(input.to(torch.int64))[0].to(torch.uint32)
    return jnp.sort(input.astype(jnp.uint32))


if jax is None:

    def solve(input, N: int):
        return _solve_impl(input, N)

else:
    solve = jax.jit(_solve_impl)
