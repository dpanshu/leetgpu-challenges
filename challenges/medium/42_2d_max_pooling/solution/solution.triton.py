import torch
import torch.nn.functional as F


def solve(
    input: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    H: int,
    W: int,
    kernel_size: int,
    stride: int,
    padding: int,
):
    input_tensor = input.view(N, C, H, W)
    result = F.max_pool2d(
        input_tensor, kernel_size=kernel_size, stride=stride, padding=padding
    )
    output.copy_(result.reshape(-1))
