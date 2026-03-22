import torch

try:
    import triton
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    triton = None


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    output.copy_(input.transpose(0, 1))
