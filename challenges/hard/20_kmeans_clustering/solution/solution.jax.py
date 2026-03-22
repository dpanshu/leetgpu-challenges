from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

import torch


def _solve_torch(
    data_x: torch.Tensor,
    data_y: torch.Tensor,
    initial_centroid_x: torch.Tensor,
    initial_centroid_y: torch.Tensor,
    sample_size: int,
    k: int,
    max_iterations: int,
):
    labels = torch.empty(sample_size, dtype=torch.int32, device=data_x.device)
    final_centroid_x = initial_centroid_x.clone()
    final_centroid_y = initial_centroid_y.clone()

    for _ in range(max_iterations):
        expanded_x = data_x.view(-1, 1) - final_centroid_x.view(1, -1)
        expanded_y = data_y.view(-1, 1) - final_centroid_y.view(1, -1)
        distances = expanded_x**2 + expanded_y**2
        labels.copy_(torch.argmin(distances, dim=1))

        for i in range(k):
            mask = labels == i
            if mask.any():
                final_centroid_x[i] = data_x[mask].mean()
                final_centroid_y[i] = data_y[mask].mean()

    return labels, final_centroid_x, final_centroid_y


def _solve_jax(
    data_x: Any,
    data_y: Any,
    initial_centroid_x: Any,
    initial_centroid_y: Any,
    sample_size: int,
    k: int,
    max_iterations: int,
):
    labels = jnp.empty((sample_size,), dtype=jnp.int32)
    final_centroid_x = jnp.array(initial_centroid_x)
    final_centroid_y = jnp.array(initial_centroid_y)

    for _ in range(max_iterations):
        expanded_x = data_x.reshape(-1, 1) - final_centroid_x.reshape(1, -1)
        expanded_y = data_y.reshape(-1, 1) - final_centroid_y.reshape(1, -1)
        distances = expanded_x**2 + expanded_y**2
        labels = jnp.argmin(distances, axis=1).astype(jnp.int32)

        for i in range(k):
            mask = labels == i
            count = jnp.sum(mask)
            safe_count = jnp.maximum(count.astype(final_centroid_x.dtype), 1.0)
            mean_x = jnp.sum(jnp.where(mask, data_x, 0.0)) / safe_count
            mean_y = jnp.sum(jnp.where(mask, data_y, 0.0)) / safe_count
            final_centroid_x = final_centroid_x.at[i].set(jnp.where(count > 0, mean_x, final_centroid_x[i]))
            final_centroid_y = final_centroid_y.at[i].set(jnp.where(count > 0, mean_y, final_centroid_y[i]))

    return labels, final_centroid_x, final_centroid_y


if jax is not None:
    solve_jax = jax.jit(
        _solve_jax,
        static_argnames=("sample_size", "k", "max_iterations"),
    )

    def solve(
        data_x: Any,
        data_y: Any,
        initial_centroid_x: Any,
        initial_centroid_y: Any,
        sample_size: int,
        k: int,
        max_iterations: int,
        labels: Any | None = None,
        final_centroid_x: Any | None = None,
        final_centroid_y: Any | None = None,
    ):
        result = solve_jax(
            data_x,
            data_y,
            initial_centroid_x,
            initial_centroid_y,
            sample_size,
            k,
            max_iterations,
        )
        if labels is not None and hasattr(labels, "copy_"):
            labels.copy_(torch.as_tensor(result[0]))
        if final_centroid_x is not None and hasattr(final_centroid_x, "copy_"):
            final_centroid_x.copy_(torch.as_tensor(result[1]))
        if final_centroid_y is not None and hasattr(final_centroid_y, "copy_"):
            final_centroid_y.copy_(torch.as_tensor(result[2]))
            return labels, final_centroid_x, final_centroid_y
        return result
else:

    def solve(
        data_x: Any,
        data_y: Any,
        initial_centroid_x: Any,
        initial_centroid_y: Any,
        sample_size: int,
        k: int,
        max_iterations: int,
        labels: Any | None = None,
        final_centroid_x: Any | None = None,
        final_centroid_y: Any | None = None,
    ):
        result = _solve_torch(
            data_x,
            data_y,
            initial_centroid_x,
            initial_centroid_y,
            sample_size,
            k,
            max_iterations,
        )
        if labels is not None and hasattr(labels, "copy_"):
            labels.copy_(result[0])
        if final_centroid_x is not None and hasattr(final_centroid_x, "copy_"):
            final_centroid_x.copy_(result[1])
        if final_centroid_y is not None and hasattr(final_centroid_y, "copy_"):
            final_centroid_y.copy_(result[2])
            return labels, final_centroid_x, final_centroid_y
        return result
