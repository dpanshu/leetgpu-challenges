import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover
    triton = None
    tl = None


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, BATCH: int, M: int, N: int, K: int):
    A_f32 = A.view(BATCH, M, K).to(torch.float32)
    B_f32 = B.view(BATCH, K, N).to(torch.float32)
    C.copy_(torch.bmm(A_f32, B_f32).to(torch.float16))
