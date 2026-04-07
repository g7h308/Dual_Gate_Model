import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import sys
import datetime  # 新增
import shutil  # 新增：用于删除文件夹
import matplotlib.pyplot as plt
from h5py.h5z import FLAG_SKIP_EDC
from torch.nn.functional import dropout
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from models.DualBranchModel import DualBranchRecurrentModel
# 引入新的加载函数
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset, load_excel_channel_data_dual

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.nn.functional import dropout, softmax # 引入 softmax 计算概率

from PCMCI import compute_causal_prior_from_channels

from tool import plot_mean_std_conf_matrix, set_seed, get_logger, close_logger, log_hyperparameters, plot_loss_curve, EarlyStopping, plot_tsne






def calculate_metrics(all_labels, all_preds, all_probs):
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # 核心：计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    try:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auc = 0.5

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm  # 返回矩阵用于后续统计
    }

# ==========================================
# 4. 训练与评估
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, A_causal_oxy, A_causal_dxy):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for oxy, dxy, labels in loader:
        oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(oxy, dxy, A_causal_oxy,A_causal_dxy)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        probs = softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return running_loss / len(loader.dataset), metrics


def evaluate(model, loader, criterion, device, A_causal_oxy, A_causal_dxy):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs = model(oxy, dxy, A_causal_oxy, A_causal_dxy)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            probs = softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return running_loss / len(loader.dataset), metrics


def evaluate_with_features(model, loader, criterion, device, A_causal_oxy, A_causal_dxy):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    all_features = []  # 新增：用于存储特征

    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            # 调用修改后的 forward
            outputs, features = model(oxy, dxy, A_causal_oxy, A_causal_dxy, return_features=True)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            probs = softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())  # 收集特征

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    return running_loss / len(loader.dataset), metrics, np.array(all_features), np.array(all_labels)

# ==========================================
# 5. 主函数 (核心修改部分)
# ==========================================

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_dims', type=int, default=64,help='隐藏向量维度')
    parser.add_argument('--head', type=int, default=2,help='头数')
    parser.add_argument('--depth', type=int, default=1,help='深度')
    parser.add_argument('--k_memory', type=int, default=10,help='记忆池长度')
    parser.add_argument('--dropout',type=float,default=0.4)
    parser.add_argument('--attn_drop',type=float,default=0.4,help='注意力drop比例')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2,help='l2正则化系数')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optim_patience', type=int, default=5,help='每隔optim_patience轮lr减半')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--exp_name', type=str, default='dual_branch')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--roi_mode', type=str, default='original',
                        choices=('original', 'full', 'hemi_4_5', 'hemi_5_4', 'three_columns', 'grid_1x3'),
                        help='original:6脑区，full：不划分  hemi_4_5:左脑4右脑5  three_columns:三等分，grid_1x3: 15个1x3大小的网格...')
    parser.add_argument('--keep_ratio', type=float, default=1.0, help='保留因果矩阵中最强连接的比例 (Top-K)')
    parser.add_argument('--chunk_size',type=int, default=10, help='滑动窗口大小')

    parser.add_argument('--disable_causal', type=bool,default=False, help='是否消融因果先验图（即不使用因果掩码，退化为全注意力）')

    parser.add_argument('--special note',type=str,default='')
    return parser.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    temp_folder_name = f"temp_{args.exp_name}_{current_time}_kfold_5metrics"
    temp_save_path = os.path.join(args.save_dir, temp_folder_name)
    final_folder_name = f"{args.exp_name}_{current_time}_kfold_metrics"
    final_save_path = os.path.join(args.save_dir, final_folder_name)

    os.makedirs(temp_save_path, exist_ok=True)
    logger = get_logger(os.path.join(temp_save_path, 'train.log'))

    try:
        logger.info(f"Start {args.k_folds}-Fold Cross Validation (Strict Subject Separation)")
        log_hyperparameters(logger, args)

        # 1. 加载原始数据 (未扩充)
        # Shape: (N_subjects, T, C, H, W)
        X_oxy_all, X_dxy_all, y_all = load_raw_data(args)

        # 还原真正的 target_len (因为 load_raw_data 里 np.delete 删掉了第0帧)
        target_len = X_oxy_all.shape[1] + 1

        # 加载未变成 grid 的原始通道数据 (双模态)
        X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)

        # 为了和 npy 严格对齐，通道数据也要删掉时间轴(axis=2)的第0帧
        X_chan_oxy_all = np.delete(X_chan_oxy_all, 0, axis=2)
        X_chan_dxy_all = np.delete(X_chan_dxy_all, 0, axis=2)

        # 定义你提供的 ROI 映射
        # 将原始通道精确映射到 15 个 1x3 的 ROI 中
        # 网格坐标系: row 0~4, col 0~8
        if args.roi_mode == 'grid_1x3':
            roi_mapping = [
                [0],  # ROI 0  (y=0, x=0~2): 包含通道 0
                [1, 2],  # ROI 1  (y=0, x=3~5): 包含通道 1, 2
                [3],  # ROI 2  (y=0, x=6~8): 包含通道 3

                [4, 5],  # ROI 3  (y=1, x=0~2): 包含通道 4, 5
                [6],  # ROI 4  (y=1, x=3~5): 包含通道 6
                [7, 8],  # ROI 5  (y=1, x=6~8): 包含通道 7, 8

                [9],  # ROI 6  (y=2, x=0~2): 包含通道 9
                [10, 11],  # ROI 7  (y=2, x=3~5): 包含通道 10, 11
                [12],  # ROI 8  (y=2, x=6~8): 包含通道 12

                [13, 14],  # ROI 9  (y=3, x=0~2): 包含通道 13, 14
                [15],  # ROI 10 (y=3, x=3~5): 包含通道 15
                [16, 17],  # ROI 11 (y=3, x=6~8): 包含通道 16, 17

                [18],  # ROI 12 (y=4, x=0~2): 包含通道 18
                [19, 20],  # ROI 13 (y=4, x=3~5): 包含通道 19, 20
                [21]  # ROI 14 (y=4, x=6~8): 包含通道 21
            ]
        else:
            # 兼容原有的映射逻辑
            roi_mapping = [
                [0, 4, 5],  # ROI 0
                [1, 2],  # ROI 1
                [3, 7, 8],  # ROI 2
                [9, 13, 14, 18],  # ROI 3
                [6, 10, 11, 15, 19, 20],  # ROI 4
                [12, 16, 17, 21]  # ROI 5
            ]

        logger.info(f"Loaded raw data. Total subjects: {len(y_all)}")

        # 2. 定义 K-Fold (基于 Subject ID 进行划分)
        kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

        fold_final_metrics = []
        all_fold_cms = []  # 新增：用于收集每一折的矩阵
        device = torch.device(args.device)

        # ==============================================================================
        # >>> 新增：仅用于观察分析，基于【全体数据】计算并打印全局因果先验图 <<<
        # ==============================================================================
        logger.info("\n>>> 开始基于全体数据计算全局 Oxy 和 Dxy 因果先验 (仅供观察分析)...")

        # 使用全体标签生成全局 Mask
        all_adhd_mask = (y_all == 0)
        all_hc_mask = (y_all == 1)

        # 1. ====== 观察全局 Oxy 因果矩阵 ======
        global_A_causal_oxy_adhd = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_oxy_all[all_adhd_mask], roi_mapping=roi_mapping
        )
        global_A_causal_oxy_hc = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_oxy_all[all_hc_mask], roi_mapping=roi_mapping
        )

        # 计算并打印 Oxy 的 3 张全局图
        global_binary_A_causal_oxy_adhd = (global_A_causal_oxy_adhd != 0).any(dim=2).float()
        global_binary_A_causal_oxy_hc = (global_A_causal_oxy_hc != 0).any(dim=2).float()
        global_binary_A_causal_oxy = (
                    global_binary_A_causal_oxy_adhd.bool() | global_binary_A_causal_oxy_hc.bool()).float()

        logger.info(f"【全局观察】binary_A_causal_oxy_adhd: \n{global_binary_A_causal_oxy_adhd}")
        logger.info(f"【全局观察】binary_A_causal_oxy_hc: \n{global_binary_A_causal_oxy_hc}")
        logger.info(f"【全局观察】binary_A_causal_oxy (并集): \n{global_binary_A_causal_oxy}")

        # 2. ====== 观察全局 Dxy 因果矩阵 ======
        global_A_causal_dxy_adhd = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_dxy_all[all_adhd_mask], roi_mapping=roi_mapping
        )
        global_A_causal_dxy_hc = compute_causal_prior_from_channels(
            fnirs_channel_data=X_chan_dxy_all[all_hc_mask], roi_mapping=roi_mapping
        )

        # 计算并打印 Dxy 的 3 张全局图
        global_binary_A_causal_dxy_adhd = (global_A_causal_dxy_adhd != 0).any(dim=2).float()
        global_binary_A_causal_dxy_hc = (global_A_causal_dxy_hc != 0).any(dim=2).float()
        global_binary_A_causal_dxy = (
                    global_binary_A_causal_dxy_adhd.bool() | global_binary_A_causal_dxy_hc.bool()).float()

        logger.info(f"【全局观察】binary_A_causal_dxy_adhd: \n{global_binary_A_causal_dxy_adhd}")
        logger.info(f"【全局观察】binary_A_causal_dxy_hc: \n{global_binary_A_causal_dxy_hc}")
        logger.info(f"【全局观察】binary_A_causal_dxy (并集): \n{global_binary_A_causal_dxy}")

        logger.info(">>> 全局因果先验图打印完毕，开始进行五折交叉验证...\n")

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

            if not args.disable_causal:
                logger.info(">>> 开始计算当前折的 Oxy 和 Dxy 因果先验...")
                X_train_chan_oxy = X_chan_oxy_all[train_idx]
                X_train_chan_dxy = X_chan_dxy_all[train_idx]

                train_adhd_mask = (y_train_raw == 0)
                train_hc_mask = (y_train_raw == 1)

                # 2. ====== 处理 Oxy 因果矩阵 ======
                # 分别计算 ADHD 和 HC 的 Oxy 矩阵
                A_causal_oxy_adhd = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_oxy[train_adhd_mask], roi_mapping=roi_mapping
                ).to(device)

                A_causal_oxy_hc = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_oxy[train_hc_mask], roi_mapping=roi_mapping
                ).to(device)

                # 取并集：只要 ADHD 或 HC 中存在非 0 的连接，我们就给它赋值为 1.0，否则为 0.0
                A_causal_oxy = torch.where((A_causal_oxy_adhd != 0) | (A_causal_oxy_hc != 0), 1.0, 0.0)


                binary_A_causal_oxy_adhd = (A_causal_oxy_adhd != 0).any(dim=2).float()
                binary_A_causal_oxy_hc = (A_causal_oxy_hc != 0).any(dim=2).float()
                logger.info(f"binary_A_causal_oxy_adhd): \n{binary_A_causal_oxy_adhd}")
                logger.info(f"binary_A_causal_oxy_hc): \n{binary_A_causal_oxy_hc}")
                binary_A_causal_oxy = (binary_A_causal_oxy_adhd.bool() | binary_A_causal_oxy_hc.bool()).float()
                logger.info(f"binary_A_causal_oxy): \n{binary_A_causal_oxy}")

                # 3. ====== 处理 Dxy 因果矩阵 ======
                # 分别计算 ADHD 和 HC 的 Dxy 矩阵
                A_causal_dxy_adhd = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_dxy[train_adhd_mask], roi_mapping=roi_mapping
                ).to(device)

                A_causal_dxy_hc = compute_causal_prior_from_channels(
                    fnirs_channel_data=X_train_chan_dxy[train_hc_mask], roi_mapping=roi_mapping
                ).to(device)

                # 取并集
                A_causal_dxy = torch.where((A_causal_dxy_adhd != 0) | (A_causal_dxy_hc != 0), 1.0, 0.0)

                binary_A_causal_dxy_adhd = (A_causal_dxy_adhd != 0).any(dim=2).float()
                binary_A_causal_dxy_hc = (A_causal_dxy_hc != 0).any(dim=2).float()
                logger.info(f"binary_A_causal_dxy_adhd): \n{binary_A_causal_dxy_adhd}")
                logger.info(f"binary_A_causal_dxy_hc): \n{binary_A_causal_dxy_hc}")
                binary_A_causal_dxy = (binary_A_causal_dxy_adhd.bool() | binary_A_causal_dxy_hc.bool()).float()
                logger.info(f"binary_A_causal_dxy): \n{binary_A_causal_dxy}")

                logger.info(">>> 双因果先验（ADHD与HC并集）计算完成！")

            else:
                logger.info(">>> [消融实验] 已禁用因果先验图，模型将退化为全空间注意力机制！")
                # 传入 None，TimeSformer 的 attn_mask 就会接收 None，从而不遮掩任何注意力
                A_causal_oxy = None
                A_causal_dxy = None

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
                roi_mode=args.roi_mode,
                keep_ratio=args.keep_ratio,
                chunk_size=args.chunk_size
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
                train_loss, t_m = train_one_epoch(model, train_loader, criterion, optimizer, device, A_causal_oxy,
                                                  A_causal_dxy)
                val_loss, v_m = evaluate(model, val_loader, criterion, device, A_causal_oxy, A_causal_dxy)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    logger.info(f"Fold {fold + 1} Epoch [{epoch + 1}/{args.epochs}] "
                                f"T_Loss: {train_loss:.4f} | T_Acc: {t_m['acc']:.4f} T_Pre: {t_m['precision']:.4f} T_Rec: {t_m['recall']:.4f} T_F1: {t_m['f1']:.4f} T_AUC: {t_m['auc']:.4f}")
                    logger.info(
                        f"V_Loss: {val_loss:.4f} | V_Acc: {v_m['acc']:.4f} V_Pre: {v_m['precision']:.4f} V_Rec: {v_m['recall']:.4f} V_F1: {v_m['f1']:.4f} V_AUC: {v_m['auc']:.4f}")

                early_stopping(val_acc=v_m['acc'], val_loss=val_loss, model=model, logger=logger)
                if early_stopping.early_stop:
                    break

            plot_loss_curve(train_losses, val_losses, temp_save_path, fold)

            # --- 验证 ---
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))
                f_loss, f_m, val_features, val_labels = evaluate_with_features(model, val_loader, criterion, device, A_causal_oxy, A_causal_dxy)
                plot_tsne(val_features, val_labels, temp_save_path, fold + 1)

                logger.info(f"Fold {fold + 1} t-SNE plot saved.")

                fold_final_metrics.append(f_m)
                all_fold_cms.append(f_m['cm'])
                logger.info(
                    f"Fold {fold + 1} BEST Result -> Acc: {f_m['acc']:.4f}, Pre: {f_m['precision']:.4f}, Rec: {f_m['recall']:.4f}, F1: {f_m['f1']:.4f}, AUC: {f_m['auc']:.4f}")
                fold_final_metrics.append(f_m)
            else:
                fold_final_metrics.append({"acc": 0, "precision": 0, "recall": 0, "f1": 0, "auc": 0})

        # --- 总结 ---

        # 1. 调用绘图函数生成均值标准差矩阵图
        if len(all_fold_cms) > 0:
            plot_mean_std_conf_matrix(all_fold_cms, temp_save_path)

        # 2. 打印原有的指标总结
        logger.info("\n" + "=" * 30)
        logger.info(f"Final {args.k_folds}-Fold CV Summary:")
        for m_name in ["acc", "precision", "recall", "f1", "auc"]:
            vals = [f[m_name] for f in fold_final_metrics]
            logger.info(f"{m_name.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
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