def solve(A, B, BATCH: int, M: int, N: int, K: int):
    try:
        import mlx.core as mx

        A_f32 = mx.array(A).reshape(BATCH, M, K).astype(mx.float32)
        B_f32 = mx.array(B).reshape(BATCH, K, N).astype(mx.float32)
        return mx.matmul(A_f32, B_f32).astype(mx.float16)
    except Exception:
        import torch

        A_f32 = torch.as_tensor(A).reshape(BATCH, M, K).to(torch.float32)
        B_f32 = torch.as_tensor(B).reshape(BATCH, K, N).to(torch.float32)
        return torch.bmm(A_f32, B_f32).to(torch.float16)
