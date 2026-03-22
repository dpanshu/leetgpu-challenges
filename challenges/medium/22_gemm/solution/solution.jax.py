import jax
import jax.numpy as jnp


@jax.jit
def solve(
    A: jax.Array, B: jax.Array, M: int, N: int, K: int, alpha: float, beta: float
) -> jax.Array:
    A_f32 = A.reshape(M, K).astype(jnp.float32)
    B_f32 = B.reshape(K, N).astype(jnp.float32)
    result = alpha * (A_f32 @ B_f32) + beta * jnp.zeros((M, N), dtype=jnp.float32)
    return result.astype(jnp.float16)
