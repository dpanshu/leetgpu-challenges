from __future__ import annotations

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - optional dependency
    jax = None
    jnp = None

import torch


def _solve_impl(signal, N: int):
    if jax is None:
        sig_c = torch.view_as_complex(signal.view(N, 2).contiguous())
        spec_c = torch.fft.fft(sig_c)
        return torch.view_as_real(spec_c).reshape(-1)

    sig_ri = jnp.reshape(signal, (N, 2))
    sig_c = sig_ri[:, 0] + 1j * sig_ri[:, 1]
    spec_c = jnp.fft.fft(sig_c)
    return jnp.reshape(jnp.stack((jnp.real(spec_c), jnp.imag(spec_c)), axis=1), (-1,))


if jax is None:

    def solve(signal, N: int):
        return _solve_impl(signal, N)

else:
    solve = jax.jit(_solve_impl, static_argnames=("N",))
