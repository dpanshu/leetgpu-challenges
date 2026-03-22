from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


if jax is not None:

    @jax.jit
    def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
        return A + B
else:

    def solve(A, B, N: int):
        return A + B
