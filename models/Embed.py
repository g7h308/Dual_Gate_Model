import torch
import torch.nn as nn
from typing import Tuple


# 假设这是你之前定义的类（保持不变）
class CustomPatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        # 定义每个patch的尺寸和在原图中的位置 [h, w, y_start, x_start]
        self.patch_definitions = [
            (2, 3, 0, 0), (1, 3, 0, 3), (2, 3, 0, 6),
            (3, 3, 2, 0), (4, 3, 1, 3), (3, 3, 2, 6)
        ]
        self.num_patches = len(self.patch_definitions)
        self.patch_projs = nn.ModuleList()
        for h, w, _, _ in self.patch_definitions:
            self.patch_projs.append(nn.Conv2d(in_channels, embed_dim, kernel_size=(h, w)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_embeddings = []
        for i, (h, w, y_start, x_start) in enumerate(self.patch_definitions):
            image_patch = x[..., y_start: y_start + h, x_start: x_start + w]
            patch_proj_output = self.patch_projs[i](image_patch)
            flattened_patch = patch_proj_output.flatten(2).transpose(1, 2)
            patch_embeddings.append(flattened_patch)
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