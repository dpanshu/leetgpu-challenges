from __future__ import annotations

from collections import deque

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

import numpy as np
import torch


def _bfs(grid_array, rows: int, cols: int, start_row: int, start_col: int, end_row: int, end_col: int):
    if start_row == end_row and start_col == end_col:
        return 0

    grid_2d = np.asarray(grid_array, dtype=np.int32).reshape(rows, cols)
    visited = np.zeros((rows, cols), dtype=bool)
    queue = deque([(start_row, start_col, 0)])
    visited[start_row, start_col] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col, dist = queue.popleft()
        if row == end_row and col == end_col:
            return dist
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if not visited[new_row, new_col] and grid_2d[new_row, new_col] == 0:
                    visited[new_row, new_col] = True
                    queue.append((new_row, new_col, dist + 1))

    return -1


def _solve_impl(grid, rows: int, cols: int, start_row: int, start_col: int, end_row: int, end_col: int):
    if jax is None:
        dist = _bfs(grid, rows, cols, start_row, start_col, end_row, end_col)
        return torch.tensor([dist], dtype=torch.int32, device=grid.device)

    dist = _bfs(jax.device_get(grid), rows, cols, start_row, start_col, end_row, end_col)
    return jnp.asarray([dist], dtype=jnp.int32)


def solve(grid, rows: int, cols: int, start_row: int, start_col: int, end_row: int, end_col: int):
    return _solve_impl(grid, rows, cols, start_row, start_col, end_row, end_col)
