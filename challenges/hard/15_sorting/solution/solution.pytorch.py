import torch


def solve(data: torch.Tensor, N: int):
    data.copy_(data.sort()[0])
