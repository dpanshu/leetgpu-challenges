import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover
    triton = None
    tl = None


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    A = A.view(BATCH, M, K)
    B = B.view(BATCH, K, N)
    C.copy_(torch.bmm(A, B))
