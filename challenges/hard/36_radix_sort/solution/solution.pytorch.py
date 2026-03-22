import torch


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    output.copy_(torch.sort(input.to(torch.int64))[0].to(torch.uint32))
