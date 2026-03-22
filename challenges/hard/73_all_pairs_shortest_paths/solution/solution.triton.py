import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(dist: torch.Tensor, output: torch.Tensor, N: int):
    d = dist.view(N, N).clone()
    for k in range(N):
        d = torch.minimum(d, d[:, k : k + 1] + d[k : k + 1, :])
    output.copy_(d.view(-1))
