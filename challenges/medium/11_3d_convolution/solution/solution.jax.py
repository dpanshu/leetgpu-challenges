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
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    if jax is None:
        input_3d = input.view(1, 1, input_depth, input_rows, input_cols)
        kernel_3d = kernel.view(1, 1, kernel_depth, kernel_rows, kernel_cols)
        return F.conv3d(input_3d, kernel_3d, padding=0).squeeze(0).squeeze(0)

    input_3d = jnp.reshape(input, (input_depth, input_rows, input_cols))
    kernel_3d = jnp.reshape(kernel, (kernel_depth, kernel_rows, kernel_cols))
    lhs = input_3d[jnp.newaxis, jnp.newaxis, :, :, :]
    rhs = kernel_3d[jnp.newaxis, jnp.newaxis, :, :, :]
    result = lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=(1, 1, 1),
        padding="VALID",
        dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
    )
    return result[0, 0]


if jax is None:
    def solve(
        input: Any,
        kernel: Any,
        input_depth: int,
        input_rows: int,
        input_cols: int,
        kernel_depth: int,
        kernel_rows: int,
        kernel_cols: int,
    ):
        return _solve_impl(
            input,
            kernel,
            input_depth,
            input_rows,
            input_cols,
            kernel_depth,
            kernel_rows,
            kernel_cols,
        )
else:
    solve = jax.jit(
        _solve_impl,
        static_argnames=(
            "input_depth",
            "input_rows",
            "input_cols",
            "kernel_depth",
            "kernel_rows",
            "kernel_cols",
        ),
    )
