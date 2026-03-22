import jax
import jax.numpy as jnp


@jax.jit
def solve(A: jax.Array, B: jax.Array, BATCH: int, M: int, N: int, K: int) -> jax.Array:
    return jnp.matmul(A.reshape(BATCH, M, K), B.reshape(BATCH, K, N))
