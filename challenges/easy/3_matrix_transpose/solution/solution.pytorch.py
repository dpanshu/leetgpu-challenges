import torch


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    output.copy_(input.transpose(0, 1))
