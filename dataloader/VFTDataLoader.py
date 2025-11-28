from sklearn.model_selection import train_test_split # 建议使用 sklearn 做索引划分
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class DualModalityDataset(Dataset):
    def __init__(self, oxy_data, dxy_data, labels):
        """
        双模态数据集
        :param oxy_data: numpy array (N, T, C) or similar
        :param dxy_data: numpy array (N, T, C) - 必须与 oxy 维度对应
        :param labels: numpy array (N,)
        """
        assert len(oxy_data) == len(dxy_data) == len(labels), "数据长度不一致！"

        self.oxy_data = torch.from_numpy(oxy_data).float()
        self.dxy_data = torch.from_numpy(dxy_data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 返回 (oxy, dxy), label
        return self.oxy_data[idx], self.dxy_data[idx], self.labels[idx]


def augment_data_odd_even(oxy, dxy, label):
    """
    数据扩充函数：奇偶采样
    输入 shape: (N, T, H, W)
    输出 shape: (2N, T/2, H, W)
    """
    # 1. 检查时间维度是否为偶数，如果是奇数，丢弃最后一帧以保证能够整除
    T = oxy.shape[1]
    if T % 2 != 0:
        print(f"Warning: Time steps {T} is odd. Trimming last frame to {T - 1} for even splitting.")
        oxy = oxy[:, :-1, ...]
        dxy = dxy[:, :-1, ...]

    # 2. 奇偶切片
    # 偶数帧: 0, 2, 4 ...
    oxy_even = oxy[:, 0::2, ...]
    dxy_even = dxy[:, 0::2, ...]

    # 奇数帧: 1, 3, 5 ...
    oxy_odd = oxy[:, 1::2, ...]
    dxy_odd = dxy[:, 1::2, ...]

    # 3. 拼接 (沿着 Batch 维度 axis=0)
    # 结果: 前 N 个是偶数采样，后 N 个是奇数采样
    oxy_aug = np.concatenate((oxy_even, oxy_odd), axis=0)
    dxy_aug = np.concatenate((dxy_even, dxy_odd), axis=0)

    # 4. 标签复制
    label_aug = np.concatenate((label, label), axis=0)

    return oxy_aug, dxy_aug, label_aug


def load_data(args):
    data_path = args.data_path

    # --- 1. 加载原始数据 (与你之前的代码一致) ---
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

    # 执行你的预处理 (删除第一帧)
    # 注意：如果原始是 1600，删掉第一帧变成 1599 (奇数)。
    # 下面的 augment_data_odd_even 会自动处理这个问题（丢弃最后一帧变成 1598）
    X_oxy = np.delete(X_oxy, 0, axis=1)
    X_dxy = np.delete(X_dxy, 0, axis=1)

    print(f"Original Data shape: {X_oxy.shape}")  # 预期 (94, 1599, 5, 9)

    # --- 2. 划分索引 (关键步骤：防止数据泄露) ---
    total_samples = len(y)
    indices = np.arange(total_samples)

    # 第一次划分：训练集 vs (验证+测试)
    # shuffle=True 保证随机性，random_state=args.seed 保证可复现
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, shuffle=True, random_state=args.seed
    )
    # 第二次划分：验证集 vs 测试集 (各占总数的 10%)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, shuffle=True, random_state=args.seed
    )

    print(f"Split sizes (Original Subjects) -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # --- 3. 根据索引提取并分别扩充 ---

    def get_augmented_set(indices):
        # 提取
        sub_oxy = X_oxy[indices]
        sub_dxy = X_dxy[indices]
        sub_y = y[indices]
        # 扩充
        return augment_data_odd_even(sub_oxy, sub_dxy, sub_y)

    # 分别处理
    train_oxy, train_dxy, train_y = get_augmented_set(train_idx)
    val_oxy, val_dxy, val_y = get_augmented_set(val_idx)
    test_oxy, test_dxy, test_y = get_augmented_set(test_idx)

    print("-" * 30)
    print("After Augmentation (Double Samples, Half Time):")
    print(f"Train set shape: {train_oxy.shape}")
    print(f"Val set shape:   {val_oxy.shape}")
    print(f"Test set shape:  {test_oxy.shape}")
    print("-" * 30)

    # --- 4. 创建 DataLoader ---
    # 注意：这时候已经是扩充后的数据了，不需要再用 random_split

    train_dataset = DualModalityDataset(train_oxy, train_dxy, train_y)
    val_dataset = DualModalityDataset(val_oxy, val_dxy, val_y)
    test_dataset = DualModalityDataset(test_oxy, test_dxy, test_y)

    # Shuffle 只需在 train_loader 中开启，验证和测试不需要
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader