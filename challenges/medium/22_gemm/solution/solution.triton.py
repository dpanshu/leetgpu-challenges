import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover
    triton = None
    tl = None


def solve(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    alpha: float,
    beta: float,
):
    A_f32 = A.view(M, K).to(torch.float32)
    B_f32 = B.view(K, N).to(torch.float32)
    C_f32 = C.view(M, N).to(torch.float32)
    C.copy_((alpha * (A_f32 @ B_f32) + beta * C_f32).to(C.dtype))
