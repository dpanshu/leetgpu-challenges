import jax
import jax.numpy as jnp


@jax.jit
def solve(A: jax.Array, B: jax.Array, BATCH: int, M: int, N: int, K: int) -> jax.Array:
    A_f32 = A.reshape(BATCH, M, K).astype(jnp.float32)
    B_f32 = B.reshape(BATCH, K, N).astype(jnp.float32)
    return jnp.matmul(A_f32, B_f32).astype(jnp.float16)
