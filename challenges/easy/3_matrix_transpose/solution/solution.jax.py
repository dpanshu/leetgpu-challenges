from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    jax = None
    jnp = None


if jax is not None:

    @jax.jit
    def solve(input: jax.Array, rows: int, cols: int) -> jax.Array:
        return input.T
else:

    def solve(input, rows: int, cols: int):
        return input.T
