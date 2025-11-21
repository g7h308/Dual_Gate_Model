import torch
import torch.nn as nn
from einops import rearrange, repeat


# 辅助模块：MLP (前馈网络)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# 核心模块：注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# 核心模块：时空 Transformer 块
class SpatioTemporalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        # 时间注意力
        self.temporal_attn = Attention(dim, heads, dim_head, dropout)
        # 空间注意力
        self.spatial_attn = Attention(dim, heads, dim_head, dropout)

        # MLP
        self.mlp = MLP(dim, mlp_dim, dim, dropout)

    def forward(self, x):
        # x shape: (b, t, n, d) - (batch, frames, num_patches, dim)
        b, t, n, d = x.shape

        # Temporal Attention
        x_temporal = rearrange(x, 'b t n d -> (b n) t d')
        x_temporal = self.temporal_attn(self.norm(x_temporal))
        x_temporal = rearrange(x_temporal, '(b n) t d -> b t n d', b=b, n=n)
        x = x + x_temporal

        # Spatial Attention
        x_spatial = rearrange(x, 'b t n d -> (b t) n d')
        x_spatial = self.spatial_attn(self.norm(x_spatial))
        x_spatial = rearrange(x_spatial, '(b t) n d -> b t n d', b=b, t=t)
        x = x + x_spatial

        # MLP
        x = x + self.mlp(self.norm(x))
        return x


# 辅助模块：门控网络
class GatingNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.gate(x)


# 最终完整模型
class DualStreamSpatioTemporalModel(nn.Module):
    def __init__(self, *,
                 num_frames, img_size, patch_size, in_channels=1, num_classes=2,
                 embed_dim=256, depth=4, heads=8, dim_head=64,
                 mlp_dim=512, feature_dim=64, dropout=0.1):
        super().__init__()

        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2

        # --- 通用参数 ---
        self.patch_size = patch_size

        # --- HBO2 流组件 ---
        self.patch_embed_hbo2 = nn.Linear(patch_dim, embed_dim)
        self.pos_embed_hbo2 = nn.Parameter(torch.randn(1, num_frames, num_patches, embed_dim))
        self.blocks_hbo2_1 = nn.ModuleList(
            [SpatioTemporalBlock(embed_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth // 2)])
        self.blocks_hbo2_2 = nn.ModuleList(
            [SpatioTemporalBlock(embed_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth - depth // 2)])

        # --- HBR 流组件 ---
        self.patch_embed_hbr = nn.Linear(patch_dim, embed_dim)
        self.pos_embed_hbr = nn.Parameter(torch.randn(1, num_frames, num_patches, embed_dim))
        self.blocks_hbr_1 = nn.ModuleList(
            [SpatioTemporalBlock(embed_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth // 2)])
        self.blocks_hbr_2 = nn.ModuleList(
            [SpatioTemporalBlock(embed_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth - depth // 2)])

        # --- 交互和融合组件 ---
        self.gate1 = GatingNetwork(embed_dim)  # HBO2 -> HBR
        self.gate2 = GatingNetwork(embed_dim)  # HBR -> HBO2

        self.linear_A1 = nn.Linear(embed_dim, feature_dim)
        self.linear_A2 = nn.Linear(embed_dim, feature_dim)
        self.linear_F1_from_hbo2 = nn.Linear(embed_dim, feature_dim)
        self.linear_F2_from_hbr = nn.Linear(embed_dim, feature_dim)

        self.mlp_classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, hbo2, hbr):
        """
        b:batch_size
        t:seq_length=frames in video
        h:height
        w:weight
        c:channels=RGB channels in video
        """
        # hbo2, hbr 初始 shape: (b, t, h, w, c)
        p = self.patch_size

        # --- Patch Embedding ---
        hbo2 = rearrange(hbo2, 'b t h w c -> b t c h w')
        hbr = rearrange(hbr, 'b t h w c -> b t c h w')

        hbo2_patches = rearrange(hbo2, 'b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=p, p2=p)
        hbr_patches = rearrange(hbr, 'b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=p, p2=p)

        hbo2_feat = self.patch_embed_hbo2(hbo2_patches)
        hbr_feat = self.patch_embed_hbr(hbr_patches)

        # --- Add Positional Embedding ---
        hbo2_feat += self.pos_embed_hbo2
        hbr_feat += self.pos_embed_hbr

        # --- 第一阶段时空处理 ---
        for blk in self.blocks_hbo2_1:
            hbo2_feat = blk(hbo2_feat)
        for blk in self.blocks_hbr_1:
            hbr_feat = blk(hbr_feat)

        # --- 第一次门控交互 (HBO2 -> HBR) ---
        gate_signal_1 = self.gate1(hbo2_feat)
        hbr_feat = hbr_feat * gate_signal_1

        # --- 第二阶段时空处理 ---
        for blk in self.blocks_hbo2_2:
            hbo2_feat = blk(hbo2_feat)
        for blk in self.blocks_hbr_2:
            hbr_feat = blk(hbr_feat)

        # --- 第二次门控交互 (HBR -> HBO2) ---
        gate_signal_2 = self.gate2(hbr_feat)
        hbo2_feat = hbo2_feat * gate_signal_2

        # --- 特征聚合 ---
        # 在时间和空间维度上取平均
        hbo2_agg = torch.mean(hbo2_feat, dim=(1, 2))  # Shape: (b, d)
        hbr_agg = torch.mean(hbr_feat, dim=(1, 2))  # Shape: (b, d)

        # --- 线性层与特征调制 ---
        A1 = self.linear_A1(hbo2_agg)
        A2 = self.linear_A2(hbr_agg)
        F1 = self.linear_F1_from_hbo2(hbo2_agg)
        F2 = self.linear_F2_from_hbr(hbr_agg)

        A1_modulated = A1 * F2
        A2_modulated = A2 * F1

        # --- 拼接与分类 ---
        fused_features = torch.cat([A1_modulated, A2_modulated], dim=1)
        output = self.mlp_classifier(fused_features)

        return output


if __name__ == '__main__':
    # --- 模拟参数 ---
    BATCH_SIZE = 4
    FRAMES = 32  # 时间帧数 (T)
    IMG_SIZE = 16  # 图像化后的高度和宽度 (H, W)
    PATCH_SIZE = 4  # 每个 Patch 的大小
    CHANNELS = 1  # 输入通道 (fNIRS=1)
    NUM_CLASSES = 2  # 分类任务的类别数

    # --- 创建模型实例 ---
    model = DualStreamSpatioTemporalModel(
        num_frames=FRAMES,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=CHANNELS,
        num_classes=NUM_CLASSES,
        embed_dim=256,  # 模型内部处理维度
        depth=6,  # Transformer Block 的总层数
        heads=8,  # 注意力头数
        mlp_dim=512,  # MLP 隐藏层维度
        feature_dim=64  # A1, A2, F1, F2 的特征维度
    )

    # --- 创建模拟输入数据 ---
    # Shape: (batch_size, frames, height, width, channels)
    dummy_hbo2 = torch.randn(BATCH_SIZE, FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)
    dummy_hbr = torch.randn(BATCH_SIZE, FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)

    print(f"输入 HBO2 Shape: {dummy_hbo2.shape}")
    print(f"输入 HBR Shape:  {dummy_hbr.shape}")

    # --- 前向传播 ---
    logits = model(dummy_hbo2, dummy_hbr)

    # --- 打印最终输出的 shape ---
    # Shape: (batch_size, num_classes)
    print(f"模型最终输出 Logits Shape: {logits.shape}")

    # --- 检查参数量 ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params / 1e6:.2f} M")