import torch
import torch.nn as nn
import torch.nn.functional as F
from .TemporalFusionModule import TemporalFusionModule
from .Timesformer import TimeSformerBlock, BIE, BIE_Concat, ConvBlock
# 引入之前写好的 Embedding 模块
from .Embed import CustomPatchEmbedding, VideoPatchEmbeddingWrapper


class DualBranchRecurrentModel(nn.Module):
    def __init__(self,
                 embed_dim=64,
                 num_heads=4,
                 depth=3,
                 k_memory=20,
                 num_classes=2,
                 chunk_size=10,drop=0.,attn_drop=0.,roi_mode='original'):  # 新增 chunk_size，对应 TimeSformer 的时间窗口
        super().__init__()

        self.depth = depth
        self.chunk_size = chunk_size
        self.embed_dim = embed_dim

        # =========================================================
        # 1. Embedding Layers (新增)
        # =========================================================
        # Branch 1: HbO2
        self.patch_embed_hbo = CustomPatchEmbedding(in_channels=1, embed_dim=embed_dim, roi_mode=roi_mode)
        num_patches = self.patch_embed_hbo.num_patches
        self.video_wrapper_hbo = VideoPatchEmbeddingWrapper(
            patch_embed_module=self.patch_embed_hbo,
            num_frames=chunk_size,  # 设为 10，因为我们在循环里每次切 10 帧
            embed_dim=embed_dim
        )

        # Branch 2: HbR (使用独立的权重，因为物理含义不同)
        self.patch_embed_hbr = CustomPatchEmbedding(in_channels=1, embed_dim=embed_dim, roi_mode=roi_mode)
        self.video_wrapper_hbr = VideoPatchEmbeddingWrapper(
            patch_embed_module=self.patch_embed_hbr,
            num_frames=chunk_size,
            embed_dim=embed_dim
        )

        # =========================================================
        # 2. Backbone 构建 (TimeSformer + BIE)
        # =========================================================
        # num_patches=6 是由 CustomPatchEmbedding 决定的
        self.hbo2_blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, num_heads, chunk_size, num_patches,drop=drop,attn_drop=attn_drop) for _ in range(depth)
        ])
        self.hbr_blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, num_heads, chunk_size, num_patches,drop=drop,attn_drop=attn_drop) for _ in range(depth)
        ])
        self.bie_layers = nn.ModuleList([
            BIE(embed_dim, num_heads) for _ in range(depth)
        ])

        # =========================================================
        # 3. Fusion Modules & Classifier
        # =========================================================
        self.fusion_hbo2 = TemporalFusionModule(embed_dim, k_memory)
        self.fusion_hbr = TemporalFusionModule(embed_dim, k_memory)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward_one_step(self, hbo2_emb, hbr_emb, A_causal_oxy=None, A_causal_dxy=None):
        """
        处理单个时间步(chunk)的前向传播
        输入形状: [B, Chunk_Size, N, D]  (例如: [B, 10, 6, 64])
        """
        x1 = hbo2_emb
        x2 = hbr_emb
        B, T, N, D = x1.shape

        # Flatten for Transformer: [B, T*N, D]
        x1 = x1.view(B, T * N, D)
        x2 = x2.view(B, T * N, D)

        # 1. 经过多层 TimeSformer + BIE 提取特征
        for i in range(self.depth):
            x1 = self.hbo2_blocks[i](x1, A_causal_oxy)
            x2 = self.hbr_blocks[i](x2, A_causal_dxy)

            x1, x2 = self.bie_layers[i](x1, x2)

        # 2. 进入 Fusion Module 与历史记忆融合
        # 注意: Fusion Module 内部通常需要处理 [B, T*N, D] 或者 [B, D]
        # 这里假设 Fusion Module 能够处理 view 后的特征
        f_t_hbo2 = self.fusion_hbo2(x1)
        f_t_hbr = self.fusion_hbr(x2)

        return f_t_hbo2, f_t_hbr

    def forward(self, hbo2_raw, hbr_raw, A_causal_oxy=None, A_causal_dxy=None):
        """
        主循环逻辑
        Input:
          hbo2_raw: [Batch, Time, H, W] (原始数据)
          hbr_raw:  [Batch, Time, H, W]
        """
        # 1. 增加 Channel 维度: [B, T, H, W] -> [B, T, 1, H, W]
        if hbo2_raw.dim() == 4:
            hbo2_raw = hbo2_raw.unsqueeze(2)
            hbr_raw = hbr_raw.unsqueeze(2)

        batch_size, time_steps, _, _, _ = hbo2_raw.shape

        # 重置记忆池
        self.fusion_hbo2.reset_memory()
        self.fusion_hbr.reset_memory()

        final_hbo2 = None
        final_hbr = None

        # 确保时间步长能被 chunk_size 整除，或者做好填充处理
        # 这里假设输入数据的长度是 chunk_size (10) 的倍数
        step = self.chunk_size

        # --- The Loop ---
        for t in range(0, time_steps, step):
            # 1. 切片 (Slice Chunk): [B, 10, 1, H, W]
            # 注意处理最后不足 10 帧的情况 (这里暂按整除处理)
            if t + step > time_steps:
                break

            chunk_hbo2 = hbo2_raw[:, t: t + step, ...]
            chunk_hbr = hbr_raw[:, t: t + step, ...]

            # 2. Embedding (Raw -> Token): [B, 10, N, D]
            # 这一步加入了 Patch Embedding 和 Position Embedding
            emb_hbo2 = self.video_wrapper_hbo(chunk_hbo2)
            emb_hbr = self.video_wrapper_hbr(chunk_hbr)

            # 3. Backbone Step
            out_hbo2, out_hbr = self.forward_one_step(
                emb_hbo2, emb_hbr,
                A_causal_oxy=A_causal_oxy,
                A_causal_dxy=A_causal_dxy
            )

            # 如果是最后一次循环，保存结果
            if t == time_steps - step:
                final_hbo2 = out_hbo2
                final_hbr = out_hbr

        # --- Classification ---
        # 此时 final_hbo2 应该包含了融合后的特征
        # 假设 output 是 [B, N, D] 或者 [B, T*N, D]，我们需要聚合

        # 简单做 Mean Pooling
        if final_hbo2.dim() == 3:  # [B, L, D]
            feat_1 = final_hbo2.mean(dim=1)
            feat_2 = final_hbr.mean(dim=1)
        else:
            feat_1 = final_hbo2
            feat_2 = final_hbr

        # 拼接
        combined_feat = torch.cat([feat_1, feat_2], dim=-1)  # [B, 2*D]

        # 分类
        logits = self.classifier(combined_feat)

        return logits


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 配置
    B = 2
    T = 20  # 假设总时长20帧
    H, W = 5, 9  # 原始网格大小
    D = 64
    CHUNK = 10  # 每次处理10帧

    # 1. 初始化模型
    model = DualBranchRecurrentModel(
        embed_dim=D,
        num_heads=4,
        depth=2,
        k_memory=5,
        num_classes=2,
        chunk_size=CHUNK
    )

    # 2. 模拟原始输入数据 [B, T, H, W]
    # 注意这里不需要预先做 patch embedding 了
    input_hbo2 = torch.randn(B, T, H, W)
    input_hbr = torch.randn(B, T, H, W)

    print(f"Input Raw Shape: {input_hbo2.shape}")

    # 3. 前向传播
    output = model(input_hbo2, input_hbr)

    print(f"Output Logits Shape: {output.shape}")  # 预期 [2, 2]

    # 验证显存/计算图连通性
    loss = output.sum()
    loss.backward()
    print("Backward pass successful.")