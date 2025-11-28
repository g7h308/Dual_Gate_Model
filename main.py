import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
import os
import logging
import sys
import time
import random
from models.DualBranchModel import DualBranchRecurrentModel


# ==========================================
# 1. 工具函数 (保持不变)
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    if save_path:
        fh = logging.FileHandler(save_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


# ==========================================
# 2. 数据加载部分 (修改核心)
# ==========================================

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


def load_data(args):
    data_path = args.data_path

    # 定义文件名
    f_adhd_oxy = os.path.join(data_path, 'ADHD_grid_oxy.npy')
    f_adhd_dxy = os.path.join(data_path, 'ADHD_grid_dxy.npy')
    f_hc_oxy = os.path.join(data_path, 'HC_grid_oxy.npy')
    f_hc_dxy = os.path.join(data_path, 'HC_grid_dxy.npy')

    # 1. 加载 Numpy 数据
    print("Loading data...")
    try:
        # ADHD Data (Label 0)
        adhd_oxy = np.load(f_adhd_oxy)
        adhd_dxy = np.load(f_adhd_dxy)
        adhd_labels = np.zeros(len(adhd_oxy))  # Label 0

        # HC Data (Label 1)
        hc_oxy = np.load(f_hc_oxy)
        hc_dxy = np.load(f_hc_dxy)
        hc_labels = np.ones(len(hc_oxy))  # Label 1

        # 检查配对样本数量是否一致
        assert len(adhd_oxy) == len(adhd_dxy), "ADHD oxy 和 dxy 样本数不匹配"
        assert len(hc_oxy) == len(hc_dxy), "HC oxy 和 dxy 样本数不匹配"

    except FileNotFoundError as e:
        print(f"Error: 找不到文件. 请确保路径正确: {e}")
        sys.exit(1)

    # 2. 拼接数据 (Concatenate)
    # 最终形式: 前半部分是ADHD，后半部分是HC (dataset split会自动打乱)
    X_oxy = np.concatenate((adhd_oxy, hc_oxy), axis=0)
    X_dxy = np.concatenate((adhd_dxy, hc_dxy), axis=0)
    y = np.concatenate((adhd_labels, hc_labels), axis=0)
    X_oxy = np.delete(X_oxy, 0, axis=1)
    X_dxy = np.delete(X_dxy, 0, axis=1)
    print(f"Total samples: {len(y)}")
    print(f"Oxy shape: {X_oxy.shape}, Dxy shape: {X_dxy.shape}")

    # 3. 创建 Dataset
    full_dataset = DualModalityDataset(X_oxy, X_dxy, y)

    # 4. 划分数据集
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    # random_split 会随机抽取索引，所以不用担心数据原本是按类排序的
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)  # 确保划分可复现
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


# ==========================================
# 3. 早停 (保持不变)
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, logger):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                # logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                pass
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, logger)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, logger):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ==========================================
# 4. 训练与验证流程 (修改核心)
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 修改：解包三个返回值 (oxy, dxy, label)
    for oxy, dxy, labels in loader:
        oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)

        optimizer.zero_grad()

        # 修改：模型接收双输入
        # 注意：这里假设你的 DualBranchRecurrentModel 的 forward 接收两个参数
        outputs = model(oxy, dxy)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # 修改：解包三个返回值
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)

            # 修改：双输入
            outputs = model(oxy, dxy)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


# ==========================================
# 5. 主函数
# ==========================================

def get_args():
    parser = argparse.ArgumentParser()
    # 只需要指定存放那4个npy文件的文件夹路径
    parser.add_argument('--data_path', type=str, default='./data/VFT', help='包含ADHD和HC数据的文件夹路径')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)  # Windows下建议0
    parser.add_argument('--hidden_dims', type=int, default=64)  # 修改：去掉 nargs='+'，避免类型混淆，除非你确定需要列表
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='dual_branch_exp')
    return parser.parse_args()


def main():
    args = get_args()
    save_path = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_path, exist_ok=True)
    logger = get_logger(os.path.join(save_path, 'train.log'))
    set_seed(args.seed)

    # 1. 加载双分支数据
    train_loader, val_loader, test_loader = load_data(args)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 2. 初始化模型
    device = torch.device(args.device)

    # 注意：请确保你的模型初始化参数与实际数据维度匹配
    # 假设输入特征维度隐含在 hidden_dims 或模型内部处理
    model = DualBranchRecurrentModel(
        embed_dim=args.hidden_dims,
        num_heads=4,
        depth=3,
        k_memory=20,
        num_classes=args.num_classes
    ).to(device)

    logger.info("Model initialized.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_model_path = os.path.join(save_path, 'best_model.pt')
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=best_model_path)

    start_time = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        logger.info(f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        early_stopping(val_loss, model, logger)
        if early_stopping.early_stop:
            logger.info("Early stopping.")
            break

    logger.info(f"Training finished in {time.time() - start_time:.2f}s")

    # 测试
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"Test Result -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")


if __name__ == '__main__':
    main()