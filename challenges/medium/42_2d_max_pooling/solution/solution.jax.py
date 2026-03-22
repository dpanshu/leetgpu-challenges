from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
except Exception:  # pragma: no cover - local fallback when JAX is unavailable
    jax = None
    jnp = None
    lax = None

import torch
import torch.nn.functional as F


def _solve_impl(
    input: Any,
    N: int,
    C: int,
    H: int,
    W: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    if jax is None:
        input_tensor = input.view(N, C, H, W)
        result = F.max_pool2d(
            input_tensor, kernel_size=kernel_size, stride=stride, padding=padding
        )
        return result.reshape(-1)

    input_tensor = jnp.reshape(input, (N, C, H, W))
    window_dims = (1, 1, kernel_size, kernel_size)
    window_strides = (1, 1, stride, stride)
    paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    result = lax.reduce_window(
        input_tensor,
        -jnp.inf,
        lax.max,
        window_dims,
        window_strides,
        paddings,
    )
    return jnp.reshape(result, (-1,))


if jax is None:
    def solve(
        input: Any,
        N: int,
        C: int,
        H: int,
        W: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        return _solve_impl(input, N, C, H, W, kernel_size, stride, padding)
else:
    solve = jax.jit(
        _solve_impl,
        static_argnames=("N", "C", "H", "W", "kernel_size", "stride", "padding"),
    )
