from __future__ import annotations

from typing import Any
import importlib.util

import torch


def _torch_impl(Q: Any, K: Any, V: Any, output: Any, M: int, d: int):
    phi_Q = torch.where(Q > 0, Q + 1, torch.exp(Q))
    phi_K = torch.where(K > 0, K + 1, torch.exp(K))
    S = phi_K.t() @ V
    z = phi_K.sum(dim=0)
    numerator = phi_Q @ S
    denominator = phi_Q @ z
    output.copy_(numerator / denominator.unsqueeze(-1))
    return output


def solve(Q: Any, K: Any, V: Any, output: Any, M: int, d: int):
    if importlib.util.find_spec("mlx.core") is None:
        return _torch_impl(Q, K, V, output, M, d)

    import mlx.core as mx

    phi_Q = mx.where(Q > 0, Q + 1, mx.exp(Q))
    phi_K = mx.where(K > 0, K + 1, mx.exp(K))
    S = phi_K.T @ V
    z = mx.sum(phi_K, axis=0)
    numerator = phi_Q @ S
    denominator = phi_Q @ z
    result = numerator / denominator[:, None]
    output[...] = result
    return output
