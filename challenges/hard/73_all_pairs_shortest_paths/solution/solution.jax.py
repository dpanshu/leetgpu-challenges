from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    lax = None

import torch


def _solve_impl(dist, N: int):
    if jax is None:
        d = dist.view(N, N).clone()
        for k in range(N):
            d = torch.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
        return d.view(-1)

    d = jnp.reshape(dist, (N, N))

    def body(k, cur):
        return jnp.minimum(cur, cur[:, k : k + 1] + cur[k : k + 1, :])

    d = lax.fori_loop(0, N, body, d)
    return jnp.reshape(d, (-1,))


if jax is None:

    def solve(dist, N: int):
        return _solve_impl(dist, N)

else:
    solve = jax.jit(_solve_impl, static_argnames=("N",))
