import torch

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    triton = None
    tl = None


def solve(signal: torch.Tensor, spectrum: torch.Tensor, N: int):
    sig_c = torch.view_as_complex(signal.view(N, 2).contiguous())
    spec_c = torch.fft.fft(sig_c)
    spectrum.copy_(torch.view_as_real(spec_c).reshape(-1))
