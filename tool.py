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
# ==========================================
# 新增: EDL 工具函数 (放置在 tool.py 末尾)
# ==========================================
import torch.nn.functional as F
import pandas as pd



PLOT_FONT_CONFIG = {
    'family': 'Times New Roman',  # 统一字体样式：新罗马
    'axis_label_size': 21,  # X轴、Y轴标签文字大小 (如 Predicted, Actual)
    'tick_label_size': 21,  # 坐标轴刻度数字、Colorbar数字的大小
    'annot_size': 30,  # 混淆矩阵内部 4 个数字的大小
    'legend_size': 18  # t-SNE 图例中文字的大小
}


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

    # 删除标题 (已注释)
    # plt.title(f'{title_prefix} t-SNE - Fold {fold_idx}')

    # --- 从字典读取配置 ---
    font_family = PLOT_FONT_CONFIG['family']
    tick_size = PLOT_FONT_CONFIG['tick_label_size']
    legend_size = PLOT_FONT_CONFIG['legend_size']

    # 调整坐标轴上的数字字体样式和大小
    plt.xticks(fontname=font_family, fontsize=tick_size)
    plt.yticks(fontname=font_family, fontsize=tick_size)

    # 调整图例字体
    plt.legend(prop={'family': font_family, 'size': legend_size})
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
    annot = np.empty_like(cm_mean).astype(str)
    rows, cols = cm_mean.shape
    for r in range(rows):
        for c in range(cols):
            annot[r, c] = f"{cm_mean[r, c]:.2f}\n±{cm_std[r, c]:.2f}"

    plt.figure(figsize=(8, 6))

    # --- 从字典读取配置 ---
    font_family = PLOT_FONT_CONFIG['family']
    annot_size = PLOT_FONT_CONFIG['annot_size']
    axis_label_size = PLOT_FONT_CONFIG['axis_label_size']
    tick_size = PLOT_FONT_CONFIG['tick_label_size']

    # 调整混淆矩阵内部图上的数字字体样式和大小
    ax = sns.heatmap(cm_mean, annot=annot, fmt="", cmap='Blues',
                     xticklabels=['ADHD', 'HC'], yticklabels=['ADHD', 'HC'],
                     annot_kws={"family": font_family, "size": annot_size})

    # 设置坐标轴标签的字体
    plt.xlabel('Predicted', fontdict={'family': font_family, 'size': axis_label_size})
    plt.ylabel('Actual', fontdict={'family': font_family, 'size': axis_label_size})

    # 设置 XY 刻度标签 ('ADHD', 'HC') 的字体
    plt.xticks(fontname=font_family, fontsize=tick_size)
    plt.yticks(fontname=font_family, fontsize=tick_size)

    # 获取热力图自带的 Colorbar 并修改上面数字的字体
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=tick_size)  # 修改 Colorbar 数字大小
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname(font_family)  # 修改 Colorbar 数字字体

    # 删除标题 (已注释)
    # plt.title('Average Confusion Matrix (Mean ± Std)')

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




def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_mse_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device):
    """EDL 的均方误差损失 + KL 散度退火"""
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    # 期望概率的均方误差
    A = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    # 预测方差
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    # KL 散度退火系数
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    alpha_tilde = y + (1 - y) * alpha
    KL = kl_divergence(alpha_tilde, num_classes, device=device)

    loss = (A + B) + annealing_coef * KL
    return loss.mean()


def plot_edl_scatter(b_list, u_list, save_path):
    """绘制 ADHD 样本的 Belief - Uncertainty 散点图"""
    plt.figure(figsize=(8, 6))

    # 使用红色散点表示 ADHD，大小适中，带白色边缘
    plt.scatter(b_list, u_list, c='#D62728', alpha=0.7, edgecolors='white', s=80, label='ADHD Samples')

    font_family = PLOT_FONT_CONFIG['family']
    plt.xlabel('Belief (b)', fontdict={'family': font_family, 'size': PLOT_FONT_CONFIG['axis_label_size']})
    plt.ylabel('Uncertainty (u)', fontdict={'family': font_family, 'size': PLOT_FONT_CONFIG['axis_label_size']})

    plt.xticks(fontname=font_family, fontsize=PLOT_FONT_CONFIG['tick_label_size'])
    plt.yticks(fontname=font_family, fontsize=PLOT_FONT_CONFIG['tick_label_size'])

    plt.legend(prop={'family': font_family, 'size': PLOT_FONT_CONFIG['legend_size']})
    plt.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'adhd_edl_scatter.png'), dpi=300)
    plt.close()


def calculate_ece(y_true, y_prob, n_bins=10):
    """计算期望校准误差 (Expected Calibration Error)"""
    # 获取预测类别的概率（置信度）和预测标签
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = predictions == y_true

    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece