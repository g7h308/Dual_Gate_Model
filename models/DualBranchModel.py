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
                 chunk_size=10,
                 drop=0.,
                 attn_drop=0.,
                 roi_mode='original',
                 keep_ratio=0.4,
                 edl_mode=False):  # 新增 chunk_size，对应 TimeSformer 的时间窗口
        super().__init__()

        self.edl_mode = edl_mode
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
            TimeSformerBlock(embed_dim, num_heads, chunk_size, num_patches,drop=drop,attn_drop=attn_drop, keep_ratio=keep_ratio) for _ in range(depth)
        ])
        self.hbr_blocks = nn.ModuleList([
            TimeSformerBlock(embed_dim, num_heads, chunk_size, num_patches,drop=drop,attn_drop=attn_drop, keep_ratio=keep_ratio) for _ in range(depth)
        ])
        self.bie_layers = nn.ModuleList([
            BIE(embed_dim, num_heads) for _ in range(depth)
        ])

        # =========================================================
        # 3. Fusion Modules & Classifier
        # =========================================================
        self.fusion_hbo2 = TemporalFusionModule(embed_dim, k_memory)
        self.fusion_hbr = TemporalFusionModule(embed_dim, k_memory)

        self.num_steps = 80

        self.classifier = nn.Sequential(
            nn.Linear(self.num_steps * embed_dim * 2, embed_dim),
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



        return x1, x2

    def forward(self, hbo2_raw, hbr_raw, A_causal_oxy=None, A_causal_dxy=None,
                return_features=False, return_step_features=False, intervene_dict=None):
        if hbo2_raw.dim() == 4:
            hbo2_raw = hbo2_raw.unsqueeze(2)
            hbr_raw = hbr_raw.unsqueeze(2)

        batch_size, time_steps, _, _, _ = hbo2_raw.shape

        # 【删】删除了重置记忆池的代码 (self.fusion_hbo2.reset_memory()等)
        # 【删】删除了 final_hbo2 = None 相关的单变量声明

        # 【增】新增用于收集每个时间步特征的列表
        step_features_hbo2 = []
        step_features_hbr = []

        step = self.chunk_size

        for t in range(0, time_steps, step):
            if t + step > time_steps:
                break

            chunk_hbo2 = hbo2_raw[:, t: t + step, ...]
            chunk_hbr = hbr_raw[:, t: t + step, ...]

            emb_hbo2 = self.video_wrapper_hbo(chunk_hbo2)
            emb_hbr = self.video_wrapper_hbr(chunk_hbr)

            out_hbo2, out_hbr = self.forward_one_step(
                emb_hbo2, emb_hbr,
                A_causal_oxy=A_causal_oxy,
                A_causal_dxy=A_causal_dxy
            )

            # 【改】保留时间因果干预逻辑，但移除对 fusion module 的依赖
            step_idx = t // step
            if intervene_dict is not None and step_idx in intervene_dict:
                hc_hbo2 = intervene_dict[step_idx]['hbo2'].to(out_hbo2.device)
                hc_hbr = intervene_dict[step_idx]['hbr'].to(out_hbr.device)

                out_hbo2 = hc_hbo2.unsqueeze(0).expand_as(out_hbo2)
                out_hbr = hc_hbr.unsqueeze(0).expand_as(out_hbr)

                # 【删】删除了将干预特征写入 self.fusion_hbo2.feature_pool[-1] 的代码，因为记忆池已消融

            # 【增】对每个 chunk 的时空 Token 维度 (dim=1) 求平均，得到当前步的表征 [B, D]
            feat_hbo2 = out_hbo2.mean(dim=1)
            feat_hbr = out_hbr.mean(dim=1)

            # 【增】将每一步的特征收集到列表中
            step_features_hbo2.append(feat_hbo2)
            step_features_hbr.append(feat_hbr)

        # 【改】将所有 step 的输出在特征维度进行拼接，形状从 len=num_steps 的 [B, D] 变为 [B, num_steps * D]
        all_steps_hbo2 = torch.cat(step_features_hbo2, dim=1)
        all_steps_hbr = torch.cat(step_features_hbr, dim=1)

        # 【改】拼接双模态特征，形状变为 [B, num_steps * D * 2]
        combined_feat = torch.cat([all_steps_hbo2, all_steps_hbr], dim=-1)

        logits = self.classifier(combined_feat)

        if self.edl_mode:
            output = F.softplus(logits)
        else:
            output = logits

        if return_features:
            return output, combined_feat

        if return_step_features:
            # 【改】直接堆叠收集到的步特征返回，形状为 [B, num_steps, D]
            return torch.stack(step_features_hbo2, dim=1), torch.stack(step_features_hbr, dim=1)

        return output


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