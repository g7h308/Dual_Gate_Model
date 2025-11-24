import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class TimeSformerBlock(nn.Module):
    """
    标准的 Transformer Encoder Block 作为 TimeSformer 的基础单元。
    这里简化使用标准的 Self-Attention，如果是 TimeSformer 的变体（如 Divided Space-Time），
    可以在这里修改 Attention 的计算方式。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch_size, patch_nums, embed_dim]
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class GatedFusion(nn.Module):
    """
    BIE 中的 Gate 模块。
    通常是一个门控机制，用于决定保留多少原始信息和多少交互信息。
    """

    def __init__(self, dim):
        super().__init__()
        # 将原始特征 X 和 交互特征 V 拼接后计算门控系数
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)  # 可选：对交互信息做一次投影

    def forward(self, x_original, x_cross):
        # x_original: 本分支的原始特征
        # x_cross: 来自另一分支的 Cross-Attention 输出

        # 计算门控系数 z (0~1)
        z = self.gate_net(torch.cat([x_original, x_cross], dim=-1))

        # 融合: z * original + (1-z) * cross (或者其他的残差形式)
        # 这里采用图示逻辑：Gate 控制输出，通常是加权和
        out = z * x_original + (1 - z) * self.proj(x_cross)
        return out


class BIE(nn.Module):
    """
    Bilateral Information Exchange (双边信息交互)
    本质是交叉注意力 + 门控。
    """

    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        # Branch 1 (HBO2) 视角的 Cross Attention: Q=X1, K=X2, V=X2
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        # Branch 2 (HBR) 视角的 Cross Attention: Q=X2, K=X1, V=X1
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout)

        self.gate_1 = GatedFusion(dim)
        self.gate_2 = GatedFusion(dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        # x1: HBO2 features [B, N, D]
        # x2: HBR features [B, N, D]

        # 1. Cross Attention
        # Branch 1 更新: Query来自X1, Key/Value来自X2
        x1_norm = self.norm1(x1)
        x2_norm = self.norm2(x2)

        # attn_out_1 是 X1 从 X2 获取的信息
        attn_out_1, _ = self.cross_attn_1(query=x1_norm, key=x2_norm, value=x2_norm)

        # Branch 2 更新: Query来自X2, Key/Value来自X1
        # attn_out_2 是 X2 从 X1 获取的信息
        attn_out_2, _ = self.cross_attn_2(query=x2_norm, key=x1_norm, value=x1_norm)

        # 2. Gating & Update
        out1 = self.gate_1(x1, attn_out_1)
        out2 = self.gate_2(x2, attn_out_2)

        return out1, out2