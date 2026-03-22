from __future__ import annotations

from typing import Any
import importlib.util
import math

import torch


def _torch_impl(Q: Any, K: Any, V: Any, output: Any, M: int, d: int):
    scores = torch.matmul(Q, K.t()) / math.sqrt(d)
    mask = torch.arange(M, device=Q.device)
    mask = mask[None, :] > mask[:, None]
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    output.copy_(attn @ V)
    return output


def solve(Q: Any, K: Any, V: Any, output: Any, M: int, d: int):
    if importlib.util.find_spec("mlx.core") is None:
        return _torch_impl(Q, K, V, output, M, d)

    import mlx.core as mx

    scores = (Q @ K.T) / math.sqrt(d)
    idxs = mx.arange(M)
    mask = idxs[None, :] > idxs[:, None]
    scores = mx.where(mask, -mx.inf, scores)
    scores = scores - mx.max(scores, axis=-1, keepdims=True)
    weights = mx.exp(scores)
    weights = weights / mx.sum(weights, axis=-1, keepdims=True)
    result = weights @ V
    output[...] = result
    return output
