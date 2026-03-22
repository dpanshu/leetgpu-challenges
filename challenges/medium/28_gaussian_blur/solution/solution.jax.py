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
    kernel: Any,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    if jax is None:
        input_2d = input.view(1, 1, input_rows, input_cols)
        kernel_2d = kernel.view(1, 1, kernel_rows, kernel_cols)
        pad_h = kernel_rows // 2
        pad_w = kernel_cols // 2
        return F.conv2d(input_2d, kernel_2d, padding=(pad_h, pad_w)).reshape(-1)

    input_2d = jnp.reshape(input, (input_rows, input_cols))
    kernel_2d = jnp.reshape(kernel, (kernel_rows, kernel_cols))
    lhs = input_2d[jnp.newaxis, jnp.newaxis, :, :]
    rhs = kernel_2d[jnp.newaxis, jnp.newaxis, :, :]
    pad_h = kernel_rows // 2
    pad_w = kernel_cols // 2
    result = lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return jnp.reshape(result, (-1,))


if jax is None:
    def solve(
        input: Any,
        kernel: Any,
        input_rows: int,
        input_cols: int,
        kernel_rows: int,
        kernel_cols: int,
    ):
        return _solve_impl(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols)
else:
    solve = jax.jit(
        _solve_impl,
        static_argnames=("input_rows", "input_cols", "kernel_rows", "kernel_cols"),
    )
