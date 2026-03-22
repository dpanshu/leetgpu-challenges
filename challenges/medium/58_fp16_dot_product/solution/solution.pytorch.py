import torch


def solve(A: torch.Tensor, B: torch.Tensor, result: torch.Tensor, N: int):
    A_f32 = A.view(N).to(torch.float32)
    B_f32 = B.view(N).to(torch.float32)
    result[0] = torch.dot(A_f32, B_f32).to(torch.float16)
