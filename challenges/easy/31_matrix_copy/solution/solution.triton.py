import torch

try:
    import triton
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    triton = None


def solve(a: torch.Tensor, b: torch.Tensor, N: int):
    b.copy_(a)
