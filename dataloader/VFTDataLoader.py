# --- START OF FILE dataloader/VFTDataLoader.py ---

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class DualModalityDataset(Dataset):
    def __init__(self, oxy_data, dxy_data, labels):
        """
        双模态数据集
        """
        # 确保输入是 Tensor
        if isinstance(oxy_data, np.ndarray):
            self.oxy_data = torch.from_numpy(oxy_data).float()
        else:
            self.oxy_data = oxy_data.float()

        if isinstance(dxy_data, np.ndarray):
            self.dxy_data = torch.from_numpy(dxy_data).float()
        else:
            self.dxy_data = dxy_data.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).long()
        else:
            self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.oxy_data[idx], self.dxy_data[idx], self.labels[idx]


def augment_data_odd_even(oxy, dxy, label):
    """
    数据扩充函数：奇偶采样
    输入 shape: (N, T, H, W)
    输出 shape: (2N, T/2, H, W)
    """
    # 1. 检查时间维度是否为偶数
    T = oxy.shape[1]
    if T % 2 != 0:
        # 默默处理，不做过多打印以免刷屏
        oxy = oxy[:, :-1, ...]
        dxy = dxy[:, :-1, ...]

    # 2. 奇偶切片
    oxy_even = oxy[:, 0::2, ...]
    dxy_even = dxy[:, 0::2, ...]

    oxy_odd = oxy[:, 1::2, ...]
    dxy_odd = dxy[:, 1::2, ...]

    # 3. 拼接
    oxy_aug = np.concatenate((oxy_even, oxy_odd), axis=0)
    dxy_aug = np.concatenate((dxy_even, dxy_odd), axis=0)

    # 4. 标签复制
    label_aug = np.concatenate((label, label), axis=0)

    return oxy_aug, dxy_aug, label_aug


def load_raw_data(args):
    """
    只负责加载原始的 numpy 数据，不进行划分和扩充。
    返回: raw_oxy, raw_dxy, raw_labels (所有样本)
    """
    data_path = args.data_path

    f_adhd_oxy = os.path.join(data_path, 'ADHD_grid_oxy.npy')
    f_adhd_dxy = os.path.join(data_path, 'ADHD_grid_dxy.npy')
    f_hc_oxy = os.path.join(data_path, 'HC_grid_oxy.npy')
    f_hc_dxy = os.path.join(data_path, 'HC_grid_dxy.npy')

    adhd_oxy = np.load(f_adhd_oxy)
    adhd_dxy = np.load(f_adhd_dxy)
    adhd_labels = np.zeros(len(adhd_oxy))

    hc_oxy = np.load(f_hc_oxy)
    hc_dxy = np.load(f_hc_dxy)
    hc_labels = np.ones(len(hc_oxy))

    # 拼接
    X_oxy = np.concatenate((adhd_oxy, hc_oxy), axis=0)
    X_dxy = np.concatenate((adhd_dxy, hc_dxy), axis=0)
    y = np.concatenate((adhd_labels, hc_labels), axis=0)

    # 统一预处理：删除第一帧 (如果需要)
    # 假设原始是 1600 -> 删掉第一帧 -> 1599 -> 后续 augment 会截断为 1598
    X_oxy = np.delete(X_oxy, 0, axis=1)
    X_dxy = np.delete(X_dxy, 0, axis=1)

    print(f"Raw Data Loaded. Shape: {X_oxy.shape}, Labels: {y.shape}")

    return X_oxy, X_dxy, y