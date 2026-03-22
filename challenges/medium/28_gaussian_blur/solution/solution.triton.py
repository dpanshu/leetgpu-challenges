import torch
import torch.nn.functional as F


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_rows: int,
    input_cols: int,
    kernel_rows: int,
    kernel_cols: int,
):
    input_2d = input.view(1, 1, input_rows, input_cols)
    kernel_2d = kernel.view(1, 1, kernel_rows, kernel_cols)
    pad_h = kernel_rows // 2
    pad_w = kernel_cols // 2
    result = F.conv2d(input_2d, kernel_2d, padding=(pad_h, pad_w))
    output.copy_(result.reshape(-1))
