import jax
import jax.numpy as jnp


@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    return jnp.reshape(jnp.dot(A.reshape(N), B.reshape(N)), (1,))
