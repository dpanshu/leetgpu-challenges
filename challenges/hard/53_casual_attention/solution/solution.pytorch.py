import math

import torch


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    scores = torch.matmul(Q, K.t()) / math.sqrt(d)
    mask = torch.arange(M, device=Q.device)
    mask = mask[None, :] > mask[:, None]
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    torch.matmul(attn, V, out=output)
