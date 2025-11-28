
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import sys
import numpy as np
# ==========================================
# 2. 数据加载部分 (保持你的逻辑)
# ==========================================

class DualModalityDataset(Dataset):
    def __init__(self, oxy_data, dxy_data, labels):
        assert len(oxy_data) == len(dxy_data) == len(labels), "数据长度不一致！"
        self.oxy_data = torch.from_numpy(oxy_data).float()
        self.dxy_data = torch.from_numpy(dxy_data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.oxy_data[idx], self.dxy_data[idx], self.labels[idx]


def load_data(args):
    data_path = args.data_path
    f_adhd_oxy = os.path.join(data_path, 'ADHD_grid_oxy.npy')
    f_adhd_dxy = os.path.join(data_path, 'ADHD_grid_dxy.npy')
    f_hc_oxy = os.path.join(data_path, 'HC_grid_oxy.npy')
    f_hc_dxy = os.path.join(data_path, 'HC_grid_dxy.npy')

    print("Loading data...")
    try:
        adhd_oxy = np.load(f_adhd_oxy)
        adhd_dxy = np.load(f_adhd_dxy)
        adhd_labels = np.zeros(len(adhd_oxy))

        hc_oxy = np.load(f_hc_oxy)
        hc_dxy = np.load(f_hc_dxy)
        hc_labels = np.ones(len(hc_oxy))
    except FileNotFoundError as e:
        print(f"Error: 找不到文件: {e}")
        sys.exit(1)

    X_oxy = np.concatenate((adhd_oxy, hc_oxy), axis=0)
    X_dxy = np.concatenate((adhd_dxy, hc_dxy), axis=0)
    y = np.concatenate((adhd_labels, hc_labels), axis=0)

    # 删除第一列（根据你的原始代码）
    X_oxy = np.delete(X_oxy, 0, axis=1)
    X_dxy = np.delete(X_dxy, 0, axis=1)

    full_dataset = DualModalityDataset(X_oxy, X_dxy, y)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader
