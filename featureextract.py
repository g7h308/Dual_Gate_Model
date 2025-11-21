import torch
import torch.nn as nn
from typing import List, Tuple


class CustomPatchEmbedding(nn.Module):
    """
    将一张图片按照预定义的、不同尺寸的patch划分模式，转换为嵌入向量序列。

    本版本不包含 [CLS] Token 和位置编码。

    输入图片尺寸: 5x9
    Patch 划分模式 (height, width, top, left):
    - p1: (2, 3) at (0, 0)
    - p2: (1, 3) at (0, 3)
    - p3: (2, 3) at (0, 6)
    - p4: (3, 3) at (2, 0)
    - p5: (4, 3) at (1, 3)
    - p6: (3, 3) at (2, 6)
    """

    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim

        # 定义每个patch的尺寸和在原图中的位置 [h, w, y_start, x_start]
        self.patch_definitions = [
            (2, 3, 0, 0),
            (1, 3, 0, 3),
            (2, 3, 0, 6),
            (3, 3, 2, 0),
            (4, 3, 1, 3),  # 调整后的位置
            (3, 3, 2, 6)
        ]

        self.num_patches = len(self.patch_definitions)

        # 为每个patch创建一个独立的卷积层
        self.patch_projs = nn.ModuleList()
        for h, w, _, _ in self.patch_definitions:
            kernel_size = (h, w)
            self.patch_projs.append(
                nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        x: 输入图片张量，形状为 [B, C, 5, 9]
        """
        B = x.shape[0]

        patch_embeddings = []
        for i, (h, w, y_start, x_start) in enumerate(self.patch_definitions):
            # 从输入 x 中精确裁剪出 patch
            image_patch = x[..., y_start: y_start + h, x_start: x_start + w]

            # 将 patch 送入对应的卷积层，输出形状: [B, embed_dim, 1, 1]
            patch_proj_output = self.patch_projs[i](image_patch)

            # 展平并调整维度，输出形状: [B, 1, embed_dim]
            flattened_patch = patch_proj_output.flatten(2).transpose(1, 2)
            patch_embeddings.append(flattened_patch)

        # 将所有patch嵌入向量拼接成一个序列
        # 输出形状: [B, 6, embed_dim]
        output_sequence = torch.cat(patch_embeddings, dim=1)

        return output_sequence


# --- 示例使用 ---
if __name__ == '__main__':
    # 定义模型参数
    img_height = 5
    img_width = 9
    embed_dim = 128  # 使用一个更合理的嵌入维度
    batch_size = 4  # 假设一个batch有4张图片

    # 实例化我们的定制嵌入层
    custom_embed_layer = CustomPatchEmbedding(in_channels=3, embed_dim=embed_dim)

    # 创建一个假的输入图片张量，尺寸必须是我们设计的 5x9
    dummy_image = torch.randn(batch_size, 3, img_height, img_width)
    print(f"输入图片形状: {dummy_image.shape}\n")

    # 将图片传入嵌入层
    output_tokens = custom_embed_layer(dummy_image)

    # 打印输出形状，这就是送入Attention之前的数据
    print(f"输出Token序列形状: {output_tokens.shape}")
    print(f"预期的形状是: [Batch_Size, Num_Patches, Embedding_Dim]")
    print(f"即: [{batch_size}, {custom_embed_layer.num_patches}, {embed_dim}]")