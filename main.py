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

from torch.nn.functional import dropout

from models.DualBranchModel import DualBranchRecurrentModel
from dataloader.VFTDataLoader import load_data


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

    # 清除之前的 handlers，防止多次调用导致重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 文件输出
    if save_path:
        fh = logging.FileHandler(save_path, mode='w')  # mode='w' 确保新建
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



# ==========================================
# 3. 早停
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
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    # --- 1. 动态生成文件夹名称 (时间戳) ---
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 临时文件夹：程序运行时数据存这里
    temp_folder_name = f"temp_{args.exp_name}_{current_time}"
    temp_save_path = os.path.join(args.save_dir, temp_folder_name)

    # 最终文件夹：程序成功结束后，重命名为此
    final_folder_name = f"{args.exp_name}_{current_time}"
    final_save_path = os.path.join(args.save_dir, final_folder_name)

    # 创建临时文件夹
    os.makedirs(temp_save_path, exist_ok=True)

    # --- 2. 初始化 Logger ---
    # Log 存放在临时文件夹中
    log_file_path = os.path.join(temp_save_path, 'train.log')
    logger = get_logger(log_file_path)

    # 使用 try...except...finally 块来管理文件夹的去留
    try:
        logger.info(f"Start Training... Temp saving to: {temp_save_path}")

        # --- 3. 记录超参数 ---
        # 这是一个新加入的功能，自动把 args 写入日志
        log_hyperparameters(logger, args)

        # 加载数据
        train_loader, val_loader, test_loader = load_data(args)
        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # 初始化模型
        device = torch.device(args.device)
        model = DualBranchRecurrentModel(
            embed_dim=args.hidden_dims,
            num_heads=args.head,
            depth=args.depth,
            k_memory=args.k_memory,
            num_classes=args.num_classes,
            drop=args.dropout,
            attn_drop=args.attn_drop
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.optim_patience)

        # EarlyStopping 保存路径设置在临时文件夹
        best_model_path = os.path.join(temp_save_path, 'best_model.pt')
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=best_model_path)

        start_time = time.time()

        # --- 训练循环 ---
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
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training finished in {time.time() - start_time:.2f}s")

        # --- 测试 ---
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            logger.info(f"Test Result -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        else:
            logger.warning("No best model saved!")

        # --- 标记成功 ---
        # 程序运行到这里没有报错，说明是正常结束
        logger.info("Process finished successfully.")

        # 必须先关闭logger，否则Windows下无法重命名文件夹（文件被占用）
        close_logger(logger)

        # 将临时文件夹重命名为正式文件夹
        if os.path.exists(temp_save_path):
            os.rename(temp_save_path, final_save_path)
            print(f"Results saved to: {final_save_path}")

    except KeyboardInterrupt:
        # 用户手动中断 (Ctrl+C)
        close_logger(logger)
        print("\nProcess Interrupted by user. Cleaning up temp files...")
        if os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)  # 递归删除文件夹
        print("Cleanup done. Nothing saved.")
        sys.exit(0)

    except Exception as e:
        # 程序发生其他错误
        close_logger(logger)
        print(f"\nAn error occurred: {e}")
        print("Cleaning up temp files due to error...")
        if os.path.exists(temp_save_path):
            shutil.rmtree(temp_save_path)
        print("Cleanup done. Nothing saved.")
        sys.exit(1)


if __name__ == '__main__':
    main()