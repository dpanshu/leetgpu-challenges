import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover
    triton = None
    tl = None


def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, n: int):
    result[0] = torch.dot(A.view(n), B.view(n))
