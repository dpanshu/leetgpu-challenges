import math

import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, N: int, d_model: int, h: int):
    d_k = d_model // h
    scale = math.sqrt(d_k)
    pieces = []
    for head in range(h):
        start = head * d_k
        stop = start + d_k
        q_h = Q[:, start:stop]
        k_h = K[:, start:stop]
        v_h = V[:, start:stop]
        scores = torch.matmul(q_h, k_h.t()) / scale
        weights = torch.softmax(scores, dim=-1)
        pieces.append(torch.matmul(weights, v_h))
    output.copy_(torch.cat(pieces, dim=-1))
