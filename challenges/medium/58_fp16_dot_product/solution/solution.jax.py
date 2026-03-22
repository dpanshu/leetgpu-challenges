import jax
import jax.numpy as jnp


@jax.jit
def solve(A: jax.Array, B: jax.Array, N: int) -> jax.Array:
    dot = jnp.dot(A.reshape(N).astype(jnp.float32), B.reshape(N).astype(jnp.float32))
    return jnp.reshape(dot.astype(jnp.float16), (1,))
