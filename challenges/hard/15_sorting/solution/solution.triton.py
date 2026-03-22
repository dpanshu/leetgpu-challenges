import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(data: torch.Tensor, N: int):
    data.copy_(torch.sort(data)[0])
