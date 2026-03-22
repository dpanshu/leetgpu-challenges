import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.sort(input.to(torch.int64))[0].to(torch.uint32))
