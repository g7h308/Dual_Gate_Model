import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class TemporalFusionModule(nn.Module):
    """
    处理特征池 (Feature Pool) 和当前特征的融合。
    """

    def __init__(self, dim, k_memory_size):
        super().__init__()
        self.k = k_memory_size
        self.dim = dim

        # 处理记忆池的 Linear 层 (如图所示，对过去 k 帧的信息进行整合)
        # 假设将 k 个特征拼接后通过 Linear 压缩或变换
        self.memory_linear = nn.Linear(dim * k_memory_size, dim)

        # 最终的 Fusion 层，将 Current Feature 和 Processed Memory 融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # 特征池：使用 deque 方便 pop(0)
        # 初始化时池子可能为空，需要处理
        self.feature_pool = deque(maxlen=k_memory_size)

    def reset_memory(self):
        """在每个新的 Batch 开始前清空记忆"""
        self.feature_pool.clear()

    def forward(self, f_t_raw):
        """
        f_t_raw: 当前时刻经过 Backbone 出来的原始特征 [B, N, D]
        """
        batch_size, num_patches, dim = f_t_raw.shape

        # 1. 如果池子没满，先用当前特征填充或者用全0填充 (策略可选)
        # 这里简单策略：如果不足 k，就复制当前特征直到填满，或者用 0 padding
        # 为了保证 tensor 操作一致性，我们假设在循环开始前已经做过初始化
        # 但为了鲁棒性，如果池子是空的，我们先填入 k 个 f_t_raw 的克隆 (冷启动)
        if len(self.feature_pool) == 0:
            for _ in range(self.k):
                self.feature_pool.append(torch.zeros_like(f_t_raw))  # 或者用 f_t_raw

        # 2. 取出池子中的 k 个特征
        # memory_feats: List of [B, N, D] -> Stack -> [B, N, k, D]
        memory_stack = torch.stack(list(self.feature_pool), dim=2)

        # 3. Linear 处理记忆 (对应图中的蓝色 Linear 条)
        # 先 flatten k 和 D 维度: [B, N, k*D]
        memory_flat = memory_stack.view(batch_size, num_patches, -1)
        memory_processed = self.memory_linear(memory_flat)  # -> [B, N, D]

        # 4. Fusion (对应图中的紫色 Fusion 块)
        # 将当前 raw 特征 与 记忆特征 结合
        f_t_fused = self.fusion_layer(torch.cat([f_t_raw, memory_processed], dim=-1))

        # 5. 更新池子 (剔除最老的，加入最新的融合特征 f_t_fused)
        # 注意：图中箭头显示放入池子的是融合后的特征 F_t
        self.feature_pool.append(f_t_fused)  # deque 会自动 popleft 如果超出 maxlen

        return f_t_fused