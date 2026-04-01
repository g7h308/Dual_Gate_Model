import torch
import torch.nn as nn
from typing import Tuple


class CustomPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 64, roi_mode: str = 'original'):
        """
        Args:
            roi_mode:
                'original': 原来的 6 个区域划分
                'full': 全脑作为一个 Patch (5x9)
                'hemi_4_5': 左脑 5x4, 右脑 5x5 (共 2 个 Patch)
                'hemi_5_4': 左脑 5x5, 右脑 5x4 (共 2 个 Patch)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.roi_mode = roi_mode

        # 定义 Patch 列表: (h, w, y_start, x_start)
        if roi_mode == 'original':
            # 原来的 6 个不规则区域
            self.patch_definitions = [
                (2, 3, 0, 0), (1, 3, 0, 3), (2, 3, 0, 6),
                (3, 3, 2, 0), (4, 3, 1, 3), (3, 3, 2, 6)
            ]
        elif roi_mode == 'full':
            # 策略一：只有一个脑区 (5x9)
            # Spatial Attention 只会对自己计算attention score，只剩 Temporal Attention 起主要作用
            self.patch_definitions = [
                (5, 9, 0, 0)
            ]
        elif roi_mode == 'hemi_4_5':
            # 策略二 (A)：左脑 5x4, 右脑 5x5
            self.patch_definitions = [
                (5, 4, 0, 0),  # Left: y=0~5, x=0~4
                (5, 5, 0, 4)  # Right: y=0~5, x=4~9
            ]
        elif roi_mode == 'hemi_5_4':
            # 策略二 (B)：左脑 5x5, 右脑 5x4
            self.patch_definitions = [
                (5, 5, 0, 0),  # Left: y=0~5, x=0~5
                (5, 4, 0, 5)  # Right: y=0~5, x=5~9
            ]
        elif roi_mode == 'three_columns':
            # 策略四：横向三等分，覆盖全高 (5x3)
            self.patch_definitions = [
                (5, 3, 0, 0),  # Left:   h=5, w=3, y=0, x=0~3
                (5, 3, 0, 3),  # Middle: h=5, w=3, y=0, x=3~6
                (5, 3, 0, 6)  # Right:  h=5, w=3, y=0, x=6~9
            ]

        elif roi_mode == 'grid_1x3':
            # 策略五：将 5x9 网格划分为 15 个 1x3 的小网格 (共15个Patch)
            self.patch_definitions = []
            for row in range(5):  # y 坐标: 0 到 4
                for col_idx in range(3):  # x 坐标被分为3段: 0~2, 3~5, 6~8
                    # 格式: (h, w, y_start, x_start)
                    self.patch_definitions.append((1, 3, row, col_idx * 3))

        else:
            raise ValueError(f"Unknown roi_mode: {roi_mode}")

        self.num_patches = len(self.patch_definitions)

        # 构建卷积投影层
        self.patch_projs = nn.ModuleList()
        for h, w, _, _ in self.patch_definitions:
            self.patch_projs.append(nn.Conv2d(in_channels, embed_dim, kernel_size=(h, w)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: [B * T, C, H, W]
        patch_embeddings = []
        for i, (h, w, y_start, x_start) in enumerate(self.patch_definitions):
            # 切片
            #image-patch:[B * T, C, H, W] -> [B * T, C, h, w]
            image_patch = x[..., y_start: y_start + h, x_start: x_start + w]
            # 卷积映射
            #[B * T, C, h, w] -> [B * T, D, 1, 1]
            patch_proj_output = self.patch_projs[i](image_patch)
            # 展平: [B, D, 1, 1] -> [B, D] -> [B, 1, D]
            # 注意: 如果 patch size 和 kernel size 一样大，输出就是 1x1
            flattened_patch = patch_proj_output.flatten(2).transpose(1, 2)
            patch_embeddings.append(flattened_patch)

        # 拼接所有 patch: [B, num_patches, D]
        return torch.cat(patch_embeddings, dim=1)


class VideoPatchEmbeddingWrapper(nn.Module):
    """
    适配器：将 5D 视频数据转换为 TimeSformer 可用的 3D 序列数据。
    包含：
    1. 维度重塑 (B, T 合并)
    2. 空间嵌入 (调用 CustomPatchEmbedding)
    3. 时间嵌入 (可学习的 Time Embedding)
    """

    def __init__(self, patch_embed_module, num_frames, embed_dim):
        super().__init__()
        self.patch_embed = patch_embed_module
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        # --- 关键点：时间位置编码 ---
        # TimeSformer 需要知道哪一帧是哪一帧。
        # 形状: [1, T, 1, D] -> 广播时会变成 [B, T, N, D]
        self.time_embed = nn.Parameter(torch.zeros(1, num_frames, 1, embed_dim))

        # 初始化参数 (通常用截断正态分布)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

    def forward(self, x):
        """
        输入 x: [Batch_Size, T, C, H, W]
        注意这里的C通道指的是RGB通道，fnirs数据当然C永远是1
        输出 out: [Batch_Size, T * N, Embed_Dim]
        """
        B, T, C, H, W = x.shape

        # 1. 【维度合并】 Merge Batch and Time
        # 将每一帧都视为独立的图片进行处理
        # [B, T, C, H, W] -> [B * T, C, H, W]
        x_merged = x.reshape(B * T, C, H, W)

        # 2. 【空间嵌入】 Process with CustomPatchEmbedding
        # 输出形状: [B * T, Num_Patches, Embed_Dim]
        # 这里的 Num_Patches 就是你定义的 6
        spatial_tokens = self.patch_embed(x_merged)

        # 获取 patch 数量 N (即 6)
        N = spatial_tokens.shape[1]

        # 3. 【维度还原】 Split Batch and Time
        # [B * T, N, D] -> [B, T, N, D]
        video_tokens = spatial_tokens.view(B, T, N, self.embed_dim)

        # 4. 【加入时间编码】 Add Time Embeddings
        # TimeSformer 极度依赖这个步骤，否则无法区分帧的顺序
        # video_tokens: [B, T, N, D] + time_embed: [1, T, 1, D]
        video_tokens = video_tokens + self.time_embed

        return video_tokens


# --- 测试代码 ---
if __name__ == '__main__':
    # 1. 定义参数
    BATCH_SIZE = 2
    FRAMES = 8  # T
    CHANNELS = 1
    HEIGHT = 5
    WIDTH = 9
    EMBED_DIM = 128

    # 2. 准备模型
    # 基础的单帧嵌入层
    base_embed = CustomPatchEmbedding(in_channels=CHANNELS, embed_dim=EMBED_DIM)
    # 视频适配器 wrapper
    video_adapter = VideoPatchEmbeddingWrapper(
        patch_embed_module=base_embed,
        num_frames=FRAMES,
        embed_dim=EMBED_DIM
    )

    # 3. 模拟输入数据 [B, T, C, H, W]
    input_video = torch.randn(BATCH_SIZE, FRAMES, CHANNELS, HEIGHT, WIDTH)
    print(f"输入视频数据形状: {input_video.shape}")

    # 4. 前向传播
    final_tokens = video_adapter(input_video)
    print(final_tokens.shape)
    # 5. 验证输出
    # 预期 Token 总数 = Frames(8) * Patches(6) = 48
    expected_seq_len = FRAMES * base_embed.num_patches

    print(f"TimeSformer 输入形状: {final_tokens.shape}")
    print(f"预期形状: [{BATCH_SIZE}, {expected_seq_len}, {EMBED_DIM}]")

    if final_tokens.shape == (BATCH_SIZE, expected_seq_len, EMBED_DIM):
        print("✅ 形状验证通过，可以直接输入 Transformer Block！")
    else:
        print("❌ 形状不匹配，请检查代码。")