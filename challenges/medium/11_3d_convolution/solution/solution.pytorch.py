import torch
import torch.nn.functional as F


def solve(
    input: torch.Tensor,
    kernel: torch.Tensor,
    output: torch.Tensor,
    input_depth: int,
    input_rows: int,
    input_cols: int,
    kernel_depth: int,
    kernel_rows: int,
    kernel_cols: int,
):
    input_3d = input.view(1, 1, input_depth, input_rows, input_cols)
    kernel_3d = kernel.view(1, 1, kernel_depth, kernel_rows, kernel_cols)
    result = F.conv3d(input_3d, kernel_3d, padding=0)
    output.copy_(result.squeeze(0).squeeze(0))
