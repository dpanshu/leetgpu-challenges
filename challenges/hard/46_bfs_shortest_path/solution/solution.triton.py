from collections import deque

import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(
    grid: torch.Tensor,
    result: torch.Tensor,
    rows: int,
    cols: int,
    start_row: int,
    start_col: int,
    end_row: int,
    end_col: int,
):
    if start_row == end_row and start_col == end_col:
        result[0] = 0
        return

    grid_2d = grid.view(rows, cols)
    visited = torch.zeros((rows, cols), dtype=torch.bool, device=grid.device)
    queue = deque([(start_row, start_col, 0)])
    visited[start_row, start_col] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col, dist = queue.popleft()
        if row == end_row and col == end_col:
            result[0] = dist
            return
        for dr, dc in directions:
            new_row = row + dr
            new_col = col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols:
                if not visited[new_row, new_col].item() and grid_2d[new_row, new_col].item() == 0:
                    visited[new_row, new_col] = True
                    queue.append((new_row, new_col, dist + 1))

    result[0] = -1
