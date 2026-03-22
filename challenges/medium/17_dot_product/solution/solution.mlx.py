def solve(A, B, N: int):
    try:
        import mlx.core as mx

        return mx.reshape(mx.sum(mx.array(A).reshape(N) * mx.array(B).reshape(N)), (1,))
    except Exception:
        import torch

        return torch.dot(torch.as_tensor(A).reshape(N), torch.as_tensor(B).reshape(N)).reshape(1)
