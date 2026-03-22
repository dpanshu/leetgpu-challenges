def solve(A, B, BATCH: int, M: int, N: int, K: int):
    try:
        import mlx.core as mx

        A = mx.array(A).reshape(BATCH, M, K)
        B = mx.array(B).reshape(BATCH, K, N)
        return mx.matmul(A, B)
    except Exception:
        import torch

        A = torch.as_tensor(A).reshape(BATCH, M, K)
        B = torch.as_tensor(B).reshape(BATCH, K, N)
        return torch.bmm(A, B)
