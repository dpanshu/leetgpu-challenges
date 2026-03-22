import math

import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    M: int,
    d: int,
    window_size: int,
):
    scores = torch.matmul(Q, K.t()) / math.sqrt(d)
    idxs = torch.arange(M, device=Q.device)
    mask = (idxs[None, :] - idxs[:, None]).abs() > window_size
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    torch.matmul(attn, V, out=output)
