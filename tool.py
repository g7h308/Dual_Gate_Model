import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import logging
import sys
import random
import json

from sklearn.manifold import TSNE


def plot_tsne(features, labels, save_path, fold_idx, title_prefix="Val Set"):
    """
    features: (N, D) 的模型输出特征
    labels: (N,) 的真实标签
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    # 假设 0 是 ADHD, 1 是 HC
    classes = ['ADHD', 'HC']
    colors = ['r', 'b']

    for i, class_name in enumerate(classes):
        mask = (labels == i)
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                    c=colors[i], label=class_name, alpha=0.6, edgecolors='w')

    plt.title(f'{title_prefix} t-SNE - Fold {fold_idx}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    img_name = f'tsne_fold_{fold_idx}.png'
    plt.savefig(os.path.join(save_path, img_name))
    plt.close()

def plot_mean_std_conf_matrix(cm_list, save_path):
    """
    cm_list: 包含 K 个混淆矩阵的列表 [cm1, cm2, ..., cmK]
    """
    cms = np.array(cm_list)  # 形状: (K, 2, 2)

    # 计算均值和标准差
    cm_mean = np.mean(cms, axis=0)
    cm_std = np.std(cms, axis=0)

    # 构建显示在方格里的文字标签 (Mean ± Std)
    # 也可以选择只显示百分比，或者 Mean(Std)
    annot = np.empty_like(cm_mean).astype(str)
    rows, cols = cm_mean.shape
    for r in range(rows):
        for c in range(cols):
            annot[r, c] = f"{cm_mean[r, c]:.2f}\n±{cm_std[r, c]:.2f}"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_mean, annot=annot, fmt="", cmap='Blues',
                xticklabels=['ADHD', 'HC'], yticklabels=['ADHD', 'HC'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Average Confusion Matrix (Mean ± Std)')
    plt.savefig(os.path.join(save_path, 'cm_mean_std.png'))
    plt.close()

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
