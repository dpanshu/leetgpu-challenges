from __future__ import annotations

from typing import Any
import importlib.util
import math

import torch


def _torch_impl(Q: Any, K: Any, V: Any, output: Any, N: int, d_model: int, h: int):
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
    return output


def solve(Q: Any, K: Any, V: Any, output: Any, N: int, d_model: int, h: int):
    if importlib.util.find_spec("mlx.core") is None:
        return _torch_impl(Q, K, V, output, N, d_model, h)

    import mlx.core as mx

    d_k = d_model // h
    scale = math.sqrt(d_k)
    pieces = []
    for head in range(h):
        start = head * d_k
        stop = start + d_k
        q_h = Q[:, start:stop]
        k_h = K[:, start:stop]
        v_h = V[:, start:stop]
        scores = (q_h @ k_h.T) / scale
        scores = scores - mx.max(scores, axis=-1, keepdims=True)
        weights = mx.exp(scores)
        weights = weights / mx.sum(weights, axis=-1, keepdims=True)
        pieces.append(weights @ v_h)
    result = mx.concatenate(pieces, axis=-1)
    output[...] = result
    return output
