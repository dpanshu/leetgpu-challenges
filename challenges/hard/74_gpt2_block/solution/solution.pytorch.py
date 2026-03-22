import math

import torch
import torch.nn.functional as F

# GPT-2 124M fixed dimensions
D = 768
H = 12
DH = D // H
FFN = 3072

O_LN1_W = 0
O_LN1_B = O_LN1_W + D
O_WQKV = O_LN1_B + D
O_BQKV = O_WQKV + D * 3 * D
O_WAPROJ = O_BQKV + 3 * D
O_BAPROJ = O_WAPROJ + D * D
O_LN2_W = O_BAPROJ + D
O_LN2_B = O_LN2_W + D
O_WFC = O_LN2_B + D
O_BFC = O_WFC + D * FFN
O_WPROJ = O_BFC + FFN
O_BPROJ = O_WPROJ + FFN * D
TOTAL_WEIGHTS = O_BPROJ + D


def solve(x: torch.Tensor, output: torch.Tensor, weights: torch.Tensor, seq_len: int):
    ln1_w = weights[O_LN1_W:O_LN1_B]
    ln1_b = weights[O_LN1_B:O_WQKV]
    W_qkv = weights[O_WQKV:O_BQKV].view(D, 3 * D)
    b_qkv = weights[O_BQKV:O_WAPROJ]
    W_attn = weights[O_WAPROJ:O_BAPROJ].view(D, D)
    b_attn = weights[O_BAPROJ:O_LN2_W]
    ln2_w = weights[O_LN2_W:O_LN2_B]
    ln2_b = weights[O_LN2_B:O_WFC]
    W_fc = weights[O_WFC:O_BFC].view(D, FFN)
    b_fc = weights[O_BFC:O_WPROJ]
    W_proj = weights[O_WPROJ:O_BPROJ].view(FFN, D)
    b_proj = weights[O_BPROJ : O_BPROJ + D]

    x_norm = F.layer_norm(x, [D], ln1_w, ln1_b, eps=1e-5)
    qkv = x_norm @ W_qkv + b_qkv
    q, k, v = qkv.split(D, dim=-1)
    q = q.view(seq_len, H, DH).transpose(0, 1)
    k = k.view(seq_len, H, DH).transpose(0, 1)
    v = v.view(seq_len, H, DH).transpose(0, 1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(DH)
    attn_weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v)

    attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, D)
    attn_proj = attn_out @ W_attn + b_attn
    hidden = x + attn_proj

    h_norm = F.layer_norm(hidden, [D], ln2_w, ln2_b, eps=1e-5)
    fc = F.gelu(h_norm @ W_fc + b_fc, approximate="tanh")
    proj = fc @ W_proj + b_proj
    output.copy_(hidden + proj)
