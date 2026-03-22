import torch


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, d: int):
    phi_Q = torch.where(Q > 0, Q + 1, torch.exp(Q))
    phi_K = torch.where(K > 0, K + 1, torch.exp(K))
    S = phi_K.t() @ V
    z = phi_K.sum(dim=0)
    numerator = phi_Q @ S
    denominator = phi_Q @ z
    output.copy_(numerator / denominator.unsqueeze(-1))
