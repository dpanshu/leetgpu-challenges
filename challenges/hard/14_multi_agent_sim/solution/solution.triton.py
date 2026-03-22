from __future__ import annotations

from typing import Any

import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(agents: torch.Tensor, agents_next: torch.Tensor, N: int):
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
    agents_next.view(N, 4).copy_(torch.cat([new_positions, new_velocities], dim=1))
