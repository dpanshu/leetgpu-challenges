import torch

try:
    import triton
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    triton = None


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    c.copy_(torch.matmul(a, b))
