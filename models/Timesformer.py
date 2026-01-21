import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimeSformerBlock(nn.Module):
    """
    Divided Space-Time Attention Block
    先做 Temporal Attention，再做 Spatial Attention
    """

    def __init__(self, dim, num_heads, num_frames, num_patches, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches

        # --- Temporal Attention (时间) ---
        self.norm1 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)

        # --- Spatial Attention (空间) ---
        self.norm2 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)

        # --- MLP ---
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # 输入 x: [B, T*N, D]
        B, Total, D = x.shape
        T = self.num_frames
        N = self.num_patches

        # === 1. Temporal Attention ===
        # 目标: [B*N, T, D] (把每个空间位置独立出来，看它随时间的变化)

        # 暂存残差
        residual = x

        x = self.norm1(x)

        # Reshape: [B, T*N, D] -> [B, T, N, D]
        x = x.view(B, T, N, D)
        # Permute: [B, T, N, D] -> [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # Reshape: [B, N, T, D] -> [B*N, T, D]
        x = x.reshape(B * N, T, D)

        # Self-Attention (sequence_len = T)
        x, _ = self.temporal_attn(x, x, x)

        # 还原形状
        # [B*N, T, D] -> [B, N, T, D]
        x = x.view(B, N, T, D)
        # [B, N, T, D] -> [B, T, N, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D] -> [B, T*N, D]
        x = x.reshape(B, T * N, D)

        # 残差连接
        x = x + residual

        # === 2. Spatial Attention ===
        # 目标: [B*T, N, D] (把每一帧独立出来，看它内部patch的关系)

        residual = x
        x = self.norm2(x)

        # Reshape: [B, T*N, D] -> [B, T, N, D]
        x = x.view(B, T, N, D)
        # Reshape: [B, T, N, D] -> [B*T, N, D]
        x = x.reshape(B * T, N, D)

        # Self-Attention (sequence_len = N)
        x, _ = self.spatial_attn(x, x, x)

        # 还原形状
        # [B*T, N, D] -> [B, T, N, D]
        x = x.view(B, T, N, D)
        # [B, T, N, D] -> [B, T*N, D]
        x = x.reshape(B, T * N, D)

        # 残差连接
        x = x + residual

        # === 3. MLP ===
        x = x + self.mlp(self.norm3(x))

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



class BIE_Concat(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        # 1. 只需要一个 Linear 层，把拼接后的 [2D] 降维回 [D]
        self.fusion = nn.Linear(dim * 2, dim)

        # 保持其他组件不动
        self.gate_1 = GatedFusion(dim)
        self.gate_2 = GatedFusion(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x1, x2):
        x1_norm = self.norm1(x1)
        x2_norm = self.norm2(x2)

        # 2. 直接 Concat，然后降维
        # cat: [B, N, D] + [B, N, D] -> [B, N, 2D]
        # fusion: [B, N, 2D] -> [B, N, D]
        merged = self.fusion(torch.cat([x1_norm, x2_norm], dim=-1))

        # 3. 把融合后的特征(merged)分别喂给两边的门控
        out1 = self.gate_1(x1, merged)
        out2 = self.gate_2(x2, merged)

        return out1, out2


# 添加到 models/Timesformer.py 文件的末尾或 TimeSformerBlock 附近

class ConvBlock(nn.Module):
    """
    用于消融实验的卷积块，完全替代 TimeSformerBlock。
    使用 1D 卷积分别在 '时间' 和 '空间' 维度上进行特征提取。
    接口参数保持一致，以便无缝替换。
    """

    def __init__(self, dim, num_heads, num_frames, num_patches, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_frames = num_frames
        self.num_patches = num_patches

        # --- Temporal Convolution (替代 Temporal Attention) ---
        self.norm1 = nn.LayerNorm(dim)
        # kernel_size=3, padding=1 保证时间维度长度不变
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(dim),  # 或者使用 GroupNorm/LayerNorm，这里用 BN 配合 Conv 比较经典
            nn.GELU()
        )

        # --- Spatial Convolution (替代 Spatial Attention) ---
        self.norm2 = nn.LayerNorm(dim)
        # kernel_size=3, padding=1 保证空间Patch数量不变
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm1d(dim),
            nn.GELU()
        )

        # --- MLP (保持不变) ---
        self.norm3 = nn.LayerNorm(dim)
        # 复用 Timesformer.py 里已有的 Mlp 类
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        # 输入 x: [B, T*N, D]
        B, Total, D = x.shape
        T = self.num_frames
        N = self.num_patches

        # === 1. Temporal Convolution ===
        # 目标: 在 Time 维度上卷积
        residual = x
        x = self.norm1(x)

        # 变换: [B, T*N, D] -> [B, T, N, D] -> [B, N, T, D] -> [B*N, D, T]
        # Conv1d 需要输入 (Batch, Channels, Length)
        x = x.view(B, T, N, D).permute(0, 2, 3, 1).reshape(B * N, D, T)

        x = self.temporal_conv(x)

        # 还原: [B*N, D, T] -> [B, N, D, T] -> [B, T, N, D] -> [B, T*N, D]
        x = x.view(B, N, D, T).permute(0, 3, 1, 2).reshape(B, Total, D)

        x = residual + x

        # === 2. Spatial Convolution ===
        # 目标: 在 Patch 维度上卷积
        residual = x
        x = self.norm2(x)

        # 变换: [B, T*N, D] -> [B, T, N, D] -> [B*T, D, N]
        x = x.view(B, T, N, D).permute(0, 1, 3, 2).reshape(B * T, D, N)

        x = self.spatial_conv(x)

        # 还原: [B*T, D, N] -> [B, T, D, N] -> [B, T, N, D] -> [B, T*N, D]
        x = x.view(B, T, D, N).permute(0, 1, 3, 2).reshape(B, Total, D)

        x = residual + x

        # === 3. MLP ===
        x = x + self.mlp(self.norm3(x))

        return x