import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

# 导入你现有的模块
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset, \
    load_excel_channel_data_dual
from models.DualBranchModel import DualBranchRecurrentModel
from PCMCI import compute_causal_prior_from_channels
from tool import set_seed, PLOT_FONT_CONFIG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--model_dir', type=str, default='./checkpoints/causal_EDL')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--hidden_dims', type=int, default=64)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--k_memory', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--roi_mode', type=str, default='original')
    parser.add_argument('--keep_ratio', type=float, default=1.0)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--disable_causal', type=bool, default=False)
    parser.add_argument('--edl_mode', type=bool, default=True)
    return parser.parse_args()


# 通道到网格坐标的映射表
coords_map = {
    0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
    4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
    9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
    13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
    18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
}

# Original 的 ROI 划分
roi_mapping = [
    [0, 4, 5],  # ROI 0
    [1, 2],  # ROI 1
    [3, 7, 8],  # ROI 2
    [9, 13, 14, 18],  # ROI 3
    [6, 10, 11, 15, 19, 20],  # ROI 4
    [12, 16, 17, 21]  # ROI 5
]


def evaluate_and_get_edl(model, loader, device, A_causal_oxy=None, A_causal_dxy=None):
    """前向传播并提取 b, u, P"""
    model.eval()
    all_b, all_u, all_P = [], [], []
    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs = model(oxy, dxy, A_causal_oxy, A_causal_dxy, return_features=False)

            # EDL 计算逻辑
            evidence = outputs
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S

            max_probs, predicted = torch.max(probs, 1)  # 期望概率 P
            uncertainty = 2 / S  # 不确定度 u (假设类别数为2)
            belief = evidence / S
            # 提取模型对预测类的 Belief b
            pred_beliefs = torch.gather(belief, 1, predicted.unsqueeze(1)).squeeze(1)

            all_b.extend(pred_beliefs.cpu().numpy().flatten())
            all_u.extend(uncertainty.cpu().numpy().flatten())
            all_P.extend(max_probs.cpu().numpy().flatten())

    return np.array(all_b), np.array(all_u), np.array(all_P)


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading data from {args.data_path}...")
    X_oxy_all, X_dxy_all, y_all = load_raw_data(args)
    target_len = X_oxy_all.shape[1] + 1
    X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)
    X_chan_oxy_all = np.delete(X_chan_oxy_all, 0, axis=2)
    X_chan_dxy_all = np.delete(X_chan_dxy_all, 0, axis=2)

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # 初始化保存所有折结果的字典
    results = {'original': {'b': [], 'u': [], 'P': []}}
    for i in range(len(roi_mapping)):
        results[f'roi_{i}'] = {'b': [], 'u': [], 'P': []}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_oxy_all)):
        print(f"\n{'=' * 20} Processing Fold {fold + 1} {'=' * 20}")

        # 1. 计算该折的因果掩码
        X_train_chan_oxy = X_chan_oxy_all[train_idx]
        X_train_chan_dxy = X_chan_dxy_all[train_idx]
        y_train_raw = y_all[train_idx]

        A_causal_oxy, A_causal_dxy = None, None
        if not args.disable_causal:
            print("  - Computing Causal Prior for this fold...")
            train_adhd_mask = (y_train_raw == 0)
            train_hc_mask = (y_train_raw == 1)

            A_oxy_adhd = compute_causal_prior_from_channels(X_train_chan_oxy[train_adhd_mask], roi_mapping).to(device)
            A_oxy_hc = compute_causal_prior_from_channels(X_train_chan_oxy[train_hc_mask], roi_mapping).to(device)
            A_causal_oxy = torch.where((A_oxy_adhd != 0) | (A_oxy_hc != 0), 1.0, 0.0)

            A_dxy_adhd = compute_causal_prior_from_channels(X_train_chan_dxy[train_adhd_mask], roi_mapping).to(device)
            A_dxy_hc = compute_causal_prior_from_channels(X_train_chan_dxy[train_hc_mask], roi_mapping).to(device)
            A_causal_dxy = torch.where((A_dxy_adhd != 0) | (A_dxy_hc != 0), 1.0, 0.0)

        # 2. 加载训练好的模型
        model_path = os.path.join(args.model_dir, f'best_model_fold_{fold}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}。请检查 --model_dir 是否正确。")

        model = DualBranchRecurrentModel(
            embed_dim=args.hidden_dims, num_heads=args.head, depth=args.depth,
            k_memory=args.k_memory, num_classes=args.num_classes, drop=args.dropout,
            attn_drop=args.attn_drop, roi_mode=args.roi_mode, keep_ratio=args.keep_ratio,
            chunk_size=args.chunk_size, edl_mode=args.edl_mode
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  - Model loaded from {model_path}")

        # 3. 提取训练集中的 HC 样本均值用于遮掩 (防数据泄露)
        train_hc_mask = (y_all[train_idx] == 1)
        mean_hc_oxy = np.mean(X_oxy_all[train_idx][train_hc_mask], axis=0)
        mean_hc_dxy = np.mean(X_dxy_all[train_idx][train_hc_mask], axis=0)

        # 4. 提取测试集中的 ADHD 样本 (Label == 0)
        val_adhd_mask = (y_all[val_idx] == 0)
        val_adhd_oxy_raw = X_oxy_all[val_idx][val_adhd_mask]
        val_adhd_dxy_raw = X_dxy_all[val_idx][val_adhd_mask]
        val_adhd_y_raw = y_all[val_idx][val_adhd_mask]

        # Baseline: 测试集 ADHD 未遮挡状态
        oxy_aug, dxy_aug, y_aug = augment_data_odd_even(val_adhd_oxy_raw, val_adhd_dxy_raw, val_adhd_y_raw)
        loader_orig = DataLoader(DualModalityDataset(oxy_aug, dxy_aug, y_aug), batch_size=args.batch_size,
                                 shuffle=False)
        b_orig, u_orig, P_orig = evaluate_and_get_edl(model, loader_orig, device, A_causal_oxy, A_causal_dxy)

        results['original']['b'].extend(b_orig)
        results['original']['u'].extend(u_orig)
        results['original']['P'].extend(P_orig)

        # Masking: 逐个遍历 6 个 ROI 执行遮挡干预
        for roi_idx, channels in enumerate(roi_mapping):
            print(f"    -> Masking ROI {roi_idx} (Channels: {channels})")
            val_oxy_masked = val_adhd_oxy_raw.copy()
            val_dxy_masked = val_adhd_dxy_raw.copy()

            for ch in channels:
                r, c = coords_map[ch]
                val_oxy_masked[:, :, r, c] = mean_hc_oxy[:, r, c]
                val_dxy_masked[:, :, r, c] = mean_hc_dxy[:, r, c]

            oxy_aug_m, dxy_aug_m, y_aug_m = augment_data_odd_even(val_oxy_masked, val_dxy_masked, val_adhd_y_raw)
            loader_m = DataLoader(DualModalityDataset(oxy_aug_m, dxy_aug_m, y_aug_m), batch_size=args.batch_size,
                                  shuffle=False)

            b_m, u_m, P_m = evaluate_and_get_edl(model, loader_m, device, A_causal_oxy, A_causal_dxy)
            results[f'roi_{roi_idx}']['b'].extend(b_m)
            results[f'roi_{roi_idx}']['u'].extend(u_m)
            results[f'roi_{roi_idx}']['P'].extend(P_m)

    # ==========================================================
    # 5. 结果保存与可视化 (重点修改的颜色搭配)
    # ==========================================================
    print("\n========== 开始生成并保存结果与图像 ==========")

    # 构建主输出目录: model_dir/causal_resultandimages
    base_out_dir = os.path.join(args.model_dir, 'causal_resultandimages')
    os.makedirs(base_out_dir, exist_ok=True)
    print(f"主输出目录已创建/存在: {base_out_dir}")

    font_family = PLOT_FONT_CONFIG['family']

    # 提取总体的 Baseline 数据
    orig_b = np.array(results['original']['b'])
    orig_u = np.array(results['original']['u'])
    orig_P = np.array(results['original']['P'])

    for roi_idx in range(len(roi_mapping)):
        # 为当前 ROI 创建专属子文件夹
        roi_dir = os.path.join(base_out_dir, f'ROI_{roi_idx}')
        os.makedirs(roi_dir, exist_ok=True)

        # 提取当前 ROI 遮蔽后的数据
        mask_b = np.array(results[f'roi_{roi_idx}']['b'])
        mask_u = np.array(results[f'roi_{roi_idx}']['u'])
        mask_P = np.array(results[f'roi_{roi_idx}']['P'])

        # ---------------- (1) 保存专属 CSV ----------------
        df = pd.DataFrame({
            'Original_b': orig_b,
            'Original_u': orig_u,
            'Original_P': orig_P,
            'Masked_b': mask_b,
            'Masked_u': mask_u,
            'Masked_P': mask_P
        })
        csv_save_path = os.path.join(roi_dir, f'result_roi_{roi_idx}.csv')
        df.to_csv(csv_save_path, index_label='Sample_Index')

        # ---------------- (2) 绘制并保存散点图 (应用新颜色方案) ----------------
        plt.figure(figsize=(9, 7))

        # 原始 ADHD 点: 修改为红色 (增加透明度和白色边框以增强清晰度)
        plt.scatter(orig_b, orig_u, c='red', alpha=0.4, edgecolors='white', label='Original ADHD', s=60)
        # 遮盖后 ADHD 点: 修改为深蓝色 (应用经典的学术蓝)
        plt.scatter(mask_b, mask_u, c='#1f77b4', alpha=0.8, edgecolors='white', label=f'Masked ROI {roi_idx}', s=80)

        # 箭头位移: 修改为中性灰色 (指示变化而不干扰整体分布观察)
        for i in range(len(orig_b)):
            plt.arrow(orig_b[i], orig_u[i], mask_b[i] - orig_b[i], mask_u[i] - orig_u[i],
                      color='gray', alpha=0.25, width=0.0015, head_width=0.01)

        plt.xlabel('Belief (b)', fontdict={'family': font_family, 'size': 18})
        plt.ylabel('Uncertainty (u)', fontdict={'family': font_family, 'size': 18})
        plt.title(f'Causal Masking Shift: ROI {roi_idx} (94 ADHD Samples)',
                  fontdict={'family': font_family, 'size': 20})

        plt.xticks(fontname=font_family, fontsize=14)
        plt.yticks(fontname=font_family, fontsize=14)
        plt.legend(prop={'family': font_family, 'size': 14})
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()

        png_save_path = os.path.join(roi_dir, f'u_b_scatter_roi_{roi_idx}.png')
        plt.savefig(png_save_path, dpi=300)
        plt.close()

        print(f"  - ROI {roi_idx} 处理完成，文件已存入: {roi_dir}")

    print("\n✅ 所有因果消融结果保存完毕！")


if __name__ == '__main__':
    main()