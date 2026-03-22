import torch

try:
    import triton
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    triton = None


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    c.copy_(a + b)
