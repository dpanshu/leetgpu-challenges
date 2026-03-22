import torch


def solve(dist: torch.Tensor, output: torch.Tensor, N: int):
    d = dist.view(N, N).clone()
    for k in range(N):
        d = torch.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    output.copy_(d.view(-1))
