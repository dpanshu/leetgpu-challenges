def solve(A, B, N: int):
    try:
        import mlx.core as mx

        dot = mx.sum(mx.array(A).reshape(N).astype(mx.float32) * mx.array(B).reshape(N).astype(mx.float32))
        return mx.reshape(dot.astype(mx.float16), (1,))
    except Exception:
        import torch

        dot = torch.dot(
            torch.as_tensor(A).reshape(N).to(torch.float32),
            torch.as_tensor(B).reshape(N).to(torch.float32),
        )
        return dot.to(torch.float16).reshape(1)
