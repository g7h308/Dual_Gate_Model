import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from TemporalFusionModule import TemporalFusionModule
from Timesformer import TimeSformerBlock, BIE

class DualBranchRecurrentModel(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_heads=4,
                 depth=4,
                 k_memory=3,
                 num_classes=2,
                 patch_nums=16):  # 假设 patch 数量固定，或者之后求平均
        super().__init__()

        self.depth = depth

        # --- Backbone 构建 ---
        # 包含若干层 TimeSformer Block 和 BIE
        # 这里的 depth 是指 (Timesformer + BIE) 重复的次数
        self.hbo2_blocks = nn.ModuleList([TimeSformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.hbr_blocks = nn.ModuleList([TimeSformerBlock(embed_dim, num_heads) for _ in range(depth)])
        self.bie_layers = nn.ModuleList([BIE(embed_dim, num_heads) for _ in range(depth)])

        # --- Fusion Modules ---
        # 两个分支各自有一个融合模块
        self.fusion_hbo2 = TemporalFusionModule(embed_dim, k_memory)
        self.fusion_hbr = TemporalFusionModule(embed_dim, k_memory)

        # --- Classification Head ---
        # 融合后的特征需要分类。假设将两个分支的最终特征拼接或相加后分类
        # 也可以对 patch 维度做 Global Average Pooling (GAP)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward_one_step(self, hbo2_emb, hbr_emb):
        """
        处理单个时间步 t 的前向传播
        hbo2_emb: [B, N, D] (已经做过 Patch Embedding)
        hbr_emb:  [B, N, D]
        """
        x1 = hbo2_emb
        x2 = hbr_emb

        # 1. 经过多层 TimeSformer + BIE 提取特征
        for i in range(self.depth):
            # 各自经过 TimeSformer Block
            x1 = self.hbo2_blocks[i](x1)
            x2 = self.hbr_blocks[i](x2)

            # 经过 BIE 进行交互
            x1, x2 = self.bie_layers[i](x1, x2)

        # 此时 x1, x2 对应图中的 F^t_{raw} (尚未融合记忆)

        # 2. 进入 Fusion Module 与历史记忆融合
        f_t_hbo2 = self.fusion_hbo2(x1)
        f_t_hbr = self.fusion_hbr(x2)

        return f_t_hbo2, f_t_hbr

    def forward(self, hbo2_seq, hbr_seq):
        """
        主循环逻辑
        Input:
          hbo2_seq: [Batch, Time, Patches, Dim]
          hbr_seq:  [Batch, Time, Patches, Dim]
        """
        batch_size, time_steps, num_patches, dim = hbo2_seq.shape

        # 每个 Batch 开始前重置记忆池
        self.fusion_hbo2.reset_memory()
        self.fusion_hbr.reset_memory()

        final_hbo2 = None
        final_hbr = None

        # --- The Loop (循环 n 次) ---
        for t in range(time_steps):
            # 取出当前帧 t 的 patch embedding
            input_t_hbo2 = hbo2_seq[:, t, :, :]
            input_t_hbr = hbr_seq[:, t, :, :]

            # 前向传播并更新记忆
            out_hbo2, out_hbr = self.forward_one_step(input_t_hbo2, input_t_hbr)

            # 如果是最后一次循环，保存结果用于分类
            if t == time_steps - 1:
                final_hbo2 = out_hbo2
                final_hbr = out_hbr

        # --- Classification ---
        # final_hbo2, final_hbr: [B, N, D]

        # 通常需要聚合 Patch 维度的信息 (Global Average Pooling)
        feat_1 = final_hbo2.mean(dim=1)  # [B, D]
        feat_2 = final_hbr.mean(dim=1)  # [B, D]

        # 拼接两个分支特征
        combined_feat = torch.cat([feat_1, feat_2], dim=-1)  # [B, 2*D]

        # 分类
        logits = self.classifier(combined_feat)  # [B, num_classes]

        return logits


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    # 假设参数
    B, T, N, D = 2, 10, 16, 64  # Batch=2, Time=10帧, Patches=16, Dim=64
    k_memory = 3
    num_classes = 5

    model = DualBranchRecurrentModel(
        embed_dim=D,
        num_heads=4,
        depth=2,
        k_memory=k_memory,
        num_classes=num_classes
    )

    # 模拟输入数据 (已经做完 Patch Embedding)
    input_hbo2 = torch.randn(B, T, N, D)
    input_hbr = torch.randn(B, T, N, D)

    # 前向传播
    output = model(input_hbo2, input_hbr)

    print(f"Input Shape: {input_hbo2.shape}")
    print(f"Output Shape: {output.shape}")  # 预期: [2, 5]
    print("模型运行成功！")