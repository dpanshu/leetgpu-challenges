def solve(A, B, M: int, N: int, K: int, alpha: float, beta: float):
    try:
        import mlx.core as mx

        A_f32 = mx.array(A).reshape(M, K).astype(mx.float32)
        B_f32 = mx.array(B).reshape(K, N).astype(mx.float32)
        result = alpha * mx.matmul(A_f32, B_f32)
        return result.astype(mx.float16)
    except Exception:
        import torch

        A_f32 = torch.as_tensor(A).reshape(M, K).to(torch.float32)
        B_f32 = torch.as_tensor(B).reshape(K, N).to(torch.float32)
        return (alpha * (A_f32 @ B_f32)).to(torch.float16)
