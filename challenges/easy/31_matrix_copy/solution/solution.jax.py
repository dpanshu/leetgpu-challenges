from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


if jax is not None:

    @jax.jit
    def solve(A: jax.Array, N: int) -> jax.Array:
        return A
else:

    def solve(A, N: int):
        return A
