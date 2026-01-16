import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import logging
import sys
import time
import random
import datetime  # 新增
import shutil  # 新增：用于删除文件夹
import json  # 新增：用于美化打印参数
import matplotlib.pyplot as plt
from torch.nn.functional import dropout
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from models.DualBranchModel import DualBranchRecurrentModel
# 引入新的加载函数
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset


# ==========================================
# 1. 工具函数
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
    """
    初始化日志
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除之前的 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 文件输出
    if save_path:
        # ============ 修改开始 ============
        # 添加 encoding='utf-8' 参数
        fh = logging.FileHandler(save_path, mode='w', encoding='utf-8')
        # ============ 修改结束 ============
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def close_logger(logger):
    """
    手动关闭日志句柄，防止删除文件夹时报错（Windows常见问题）
    """
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)


def log_hyperparameters(logger, args):
    """
    将所有超参数打印并保存到日志中
    """
    logger.info("=" * 30)
    logger.info("Current Configuration (Hyperparameters):")
    # 将 args 转为字典并美化打印
    args_dict = vars(args)
    # json.dumps 用于格式化字符串
    logger.info(json.dumps(args_dict, indent=4))
    logger.info("=" * 30)


def plot_loss_curve(train_losses, val_losses, save_path, fold_idx):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red', linestyle='--')
    plt.title(f'Fold {fold_idx} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f'loss_curve_fold_{fold_idx}.png'))
    plt.close()


# ==========================================
# 3. 早停
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): 多少个 epoch 没有提升（Acc没变大 且 Loss没变小）就停止
            verbose (bool): 是否打印日志
            path (str): 模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.early_stop = False

        # 记录最佳指标
        self.best_acc = -np.Inf
        self.best_loss = np.Inf

    def __call__(self, val_acc, val_loss, model, logger):
        """
        逻辑：
        1. 如果当前 Acc > 历史最佳 Acc：保存模型 (更新最佳 Acc 和 Loss)
        2. 如果当前 Acc == 历史最佳 Acc：
             如果当前 Loss < 历史最佳 Loss：保存模型 (更新最佳 Loss)
             否则：计数器 +1
        3. 如果当前 Acc < 历史最佳 Acc：计数器 +1
        """

        # 情况 1: 准确率创新高
        if val_acc > self.best_acc:
            if self.verbose:
                logger.info(f'Validation Acc increased ({self.best_acc:.4f} --> {val_acc:.4f}). Saving model...')
            self.best_acc = val_acc
            self.best_loss = val_loss  # 同时更新对应的 loss
            self.save_checkpoint(model)
            self.counter = 0  # 重置计数器

        # 情况 2: 准确率持平，看 Loss 是否降低
        elif val_acc == self.best_acc:
            if val_loss < self.best_loss:
                if self.verbose:
                    logger.info(
                        f'Acc same ({self.best_acc:.4f}), but Loss decreased ({self.best_loss:.4f} --> {val_loss:.4f}). Saving model...')
                self.best_loss = val_loss
                self.save_checkpoint(model)
                self.counter = 0  # 重置计数器
            else:
                self.counter += 1
                if self.verbose:
                    logger.info(
                        f'EarlyStopping counter: {self.counter} out of {self.patience} (Acc same, Loss not improved)')

        # 情况 3: 准确率下降
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} (Acc decreased)')

        # 检查是否触发早停
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)


# ==========================================
# 4. 训练与评估
# ==========================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for oxy, dxy, labels in loader:
        oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(oxy, dxy)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs = model(oxy, dxy)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


# ==========================================
# 5. 主函数 (核心修改部分)
# ==========================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_dims', type=int, default=64)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--k_memory', type=int, default=10)
    parser.add_argument('--dropout',type=float,default=0.4)
    parser.add_argument('--attn_drop',type=float,default=0.4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--optim_patience', type=int, default=5,help='每隔optim_patience轮lr减半')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='dual_branch')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--roi_mode',type=str, default='original',choices=('original', 'full', 'hemi_4_5', 'hemi_5_4','three_columns'),
                        help='original:6脑区，full：不划分  hemi_4_5:左脑4右脑5  three_columns:三等分')
    parser.add_argument('--special note',type=str,default='')
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_folder_name = f"temp_{args.exp_name}_{current_time}_kfold"
    temp_save_path = os.path.join(args.save_dir, temp_folder_name)
    final_folder_name = f"{args.exp_name}_{current_time}_kfold"
    final_save_path = os.path.join(args.save_dir, final_folder_name)

    os.makedirs(temp_save_path, exist_ok=True)
    logger = get_logger(os.path.join(temp_save_path, 'train.log'))

    try:
        logger.info(f"Start {args.k_folds}-Fold Cross Validation (Strict Subject Separation)")
        log_hyperparameters(logger, args)

        # 1. 加载原始数据 (未扩充)
        # Shape: (N_subjects, T, C, H, W)
        X_oxy_all, X_dxy_all, y_all = load_raw_data(args)

        logger.info(f"Loaded raw data. Total subjects: {len(y_all)}")

        # 2. 定义 K-Fold (基于 Subject ID 进行划分)
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

        fold_results = []
        device = torch.device(args.device)

        # 3. K-Fold 循环
        # split 的输入是 range(N_subjects)，保证同一个人的数据要么都在训练，要么都在验证
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_oxy_all)):
            logger.info(f"\n{'=' * 20} Fold [{fold + 1}/{args.k_folds}] {'=' * 20}")
            logger.info(f"Train subjects: {len(train_idx)}, Val subjects: {len(val_idx)}")

            # --- 关键步骤：先根据索引切分 ---
            X_train_oxy_raw = X_oxy_all[train_idx]
            X_train_dxy_raw = X_dxy_all[train_idx]
            y_train_raw = y_all[train_idx]

            X_val_oxy_raw = X_oxy_all[val_idx]
            X_val_dxy_raw = X_dxy_all[val_idx]
            y_val_raw = y_all[val_idx]

            # --- 关键步骤：然后在各自集合内独立进行扩充 ---
            # 这样 Train 里的扩充样本只来自 Train Subject，Val 同理
            train_oxy_aug, train_dxy_aug, train_y_aug = augment_data_odd_even(X_train_oxy_raw, X_train_dxy_raw,
                                                                              y_train_raw)
            val_oxy_aug, val_dxy_aug, val_y_aug = augment_data_odd_even(X_val_oxy_raw, X_val_dxy_raw, y_val_raw)

            logger.info(f"Augmented Train Samples: {len(train_y_aug)} | Augmented Val Samples: {len(val_y_aug)}")

            # --- 构建 Dataset 和 DataLoader ---
            train_dataset = DualModalityDataset(train_oxy_aug, train_dxy_aug, train_y_aug)
            val_dataset = DualModalityDataset(val_oxy_aug, val_dxy_aug, val_y_aug)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=args.num_workers)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

            # --- 模型初始化 (每折重置) ---
            model = DualBranchRecurrentModel(
                embed_dim=args.hidden_dims,
                num_heads=args.head,
                depth=args.depth,
                k_memory=args.k_memory,
                num_classes=args.num_classes,
                drop=args.dropout,
                attn_drop=args.attn_drop,
                roi_mode=args.roi_mode
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                             patience=args.optim_patience)

            best_model_path = os.path.join(temp_save_path, f'best_model_fold_{fold}.pt')
            early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=best_model_path)

            train_losses = []
            val_losses = []

            # --- 训练 ---
            for epoch in range(args.epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)

                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"Fold {fold + 1} Epoch [{epoch + 1}/{args.epochs}] "
                                f"T_Loss: {train_loss:.4f} T_Acc: {train_acc:.4f} | "
                                f"V_Loss: {val_loss:.4f} V_Acc: {val_acc:.4f}")

                early_stopping(val_acc=val_acc, val_loss=val_loss, model=model, logger=logger)

                if early_stopping.early_stop:
                    logger.info(f"Fold {fold + 1} Early stopping triggered.")
                    break

            plot_loss_curve(train_losses, val_losses, temp_save_path, fold)

            # --- 验证 ---
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                final_loss, final_acc = evaluate(model, val_loader, criterion, device)
                logger.info(
                    f"Fold {fold + 1} BEST Model (High Acc, Low Loss) -> Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")
                fold_results.append(final_acc)
            else:
                fold_results.append(0.0)

        # --- 总结 ---
        avg_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        logger.info("\n" + "=" * 30)
        logger.info(f"Final 5-Fold CV Results:")
        for i, acc in enumerate(fold_results):
            logger.info(f"Fold {i + 1}: {acc:.4f}")
        logger.info(f"Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
        logger.info("=" * 30)

        close_logger(logger)
        if os.path.exists(temp_save_path):
            os.rename(temp_save_path, final_save_path)
            print(f"Saved to: {final_save_path}")

    except Exception as e:
        close_logger(logger)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()