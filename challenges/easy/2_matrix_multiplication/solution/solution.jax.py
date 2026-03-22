from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


if jax is not None:

    @jax.jit
    def solve(A: jax.Array, B: jax.Array, M: int, N: int, K: int) -> jax.Array:
        return A @ B
else:

    def solve(A, B, M: int, N: int, K: int):
        return A @ B
