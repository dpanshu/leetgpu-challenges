from __future__ import annotations

from typing import Any

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

import torch


def _solve_torch(agents: torch.Tensor, N: int) -> torch.Tensor:
    agents_reshaped = agents.view(N, 4)
    positions = agents_reshaped[:, :2]
    velocities = agents_reshaped[:, 2:]

    diff = positions.unsqueeze(1) - positions.unsqueeze(0)
    dist_sq = (diff**2).sum(dim=2)
    dist_sq.fill_diagonal_(26.0)

    neighbor_mask = dist_sq < 25.0
    sum_velocities = neighbor_mask.float() @ velocities
    neighbor_counts = neighbor_mask.sum(dim=1, keepdim=True)

    avg_velocities = torch.empty_like(velocities)
    nonzero_mask = neighbor_counts[:, 0] > 0
    avg_velocities[nonzero_mask] = sum_velocities[nonzero_mask] / neighbor_counts[nonzero_mask]
    avg_velocities[~nonzero_mask] = velocities[~nonzero_mask]

    new_velocities = velocities + 0.05 * (avg_velocities - velocities)
    new_positions = positions + new_velocities
    return torch.cat([new_positions, new_velocities], dim=1).reshape(-1)


def _solve_jax(agents: Any, N: int):
    agents_reshaped = jnp.reshape(agents, (N, 4))
    positions = agents_reshaped[:, :2]
    velocities = agents_reshaped[:, 2:]

    diff = positions[:, None, :] - positions[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=2)
    idx = jnp.arange(N)
    dist_sq = dist_sq.at[idx, idx].set(26.0)

    neighbor_mask = dist_sq < 25.0
    sum_velocities = neighbor_mask.astype(velocities.dtype) @ velocities
    neighbor_counts = jnp.sum(neighbor_mask, axis=1, keepdims=True)
    safe_counts = jnp.maximum(neighbor_counts.astype(velocities.dtype), 1.0)
    avg_velocities = sum_velocities / safe_counts
    avg_velocities = jnp.where(neighbor_counts > 0, avg_velocities, velocities)

    new_velocities = velocities + 0.05 * (avg_velocities - velocities)
    new_positions = positions + new_velocities
    return jnp.reshape(jnp.concatenate([new_positions, new_velocities], axis=1), (-1,))


if jax is not None:
    solve_jax = jax.jit(_solve_jax, static_argnames=("N",))

    def solve(agents: Any, N: int, agents_next: Any | None = None):
        result = solve_jax(agents, N)
        if agents_next is not None and hasattr(agents_next, "copy_"):
            agents_next.copy_(torch.as_tensor(result))
            return agents_next
        return result
else:

    def solve(agents: Any, N: int, agents_next: Any | None = None):
        result = _solve_torch(agents, N)
        if agents_next is not None and hasattr(agents_next, "copy_"):
            agents_next.copy_(result)
            return agents_next
        return result
