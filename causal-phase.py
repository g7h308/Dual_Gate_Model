import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# 导入你现有的模块
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset, \
    load_excel_channel_data_dual
from models.DualBranchModel import DualBranchRecurrentModel
from PCMCI import compute_causal_prior_from_channels
from tool import set_seed, PLOT_FONT_CONFIG


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--model_dir', type=str, default='./checkpoints/causal_EDL')  # 确保指向你用原始main.py训练保存的模型文件夹
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


# ROI 划分
roi_mapping = [
    [0, 4, 5], [1, 2], [3, 7, 8], [9, 13, 14, 18], [6, 10, 11, 15, 19, 20], [12, 16, 17, 21]
]


def calculate_ece(confidences, accuracies, n_bins=10):
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        if i == 0:
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        else:
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


def evaluate_and_get_edl(model, loader, device, A_causal_oxy=None, A_causal_dxy=None):
    model.eval()
    all_b, all_u, all_P = [], [], []
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            outputs = model(oxy, dxy, A_causal_oxy, A_causal_dxy, return_features=False)

            evidence = outputs
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S

            max_probs, predicted = torch.max(probs, 1)
            uncertainty = 2 / S
            belief = evidence / S
            pred_beliefs = torch.gather(belief, 1, predicted.unsqueeze(1)).squeeze(1)

            all_b.extend(pred_beliefs.cpu().numpy().flatten())
            all_u.extend(uncertainty.cpu().numpy().flatten())
            all_P.extend(max_probs.cpu().numpy().flatten())

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    return np.array(all_b), np.array(all_u), np.array(all_P), np.array(all_preds), np.array(all_labels), np.array(
        all_probs)


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Loading data from {args.data_path}...")
    # 1. 加载所有真实的原始数据
    X_oxy_orig, X_dxy_orig, y_all = load_raw_data(args)
    target_len = X_oxy_orig.shape[1] + 1
    X_chan_oxy_orig, X_chan_dxy_orig = load_excel_channel_data_dual(args.data_path, target_len=target_len)
    X_chan_oxy_orig = np.delete(X_chan_oxy_orig, 0, axis=2)
    X_chan_dxy_orig = np.delete(X_chan_dxy_orig, 0, axis=2)

    # 2. 加载修正后的 ADHD 数据并构造出一套 "被干预的平行宇宙数据"
    print(">>> 正在加载修正后(健康化)的 ADHD 数据集...")
    adhd_mask = (y_all == 0)

    corr_grid_oxy = np.load(os.path.join(args.data_path, 'ADHD_grid_oxy_corrected.npy'))
    corr_grid_dxy = np.load(os.path.join(args.data_path, 'ADHD_grid_dxy_corrected.npy'))

    X_oxy_corr = X_oxy_orig.copy()
    X_dxy_corr = X_dxy_orig.copy()
    X_oxy_corr[adhd_mask] = corr_grid_oxy
    X_dxy_corr[adhd_mask] = corr_grid_dxy

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    results = {
        'original': {'b': [], 'u': [], 'P': [], 'correct': [], 'fold_metrics': []},
        'corrected': {'b': [], 'u': [], 'P': [], 'correct': [], 'fold_metrics': []}
    }

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_oxy_orig)):
        print(f"\n{'=' * 20} Processing Fold {fold + 1} {'=' * 20}")

        # --- 核心：因果图必须基于真实的原始训练集生成，因为模型是看着真实因果图长大的 ---
        A_causal_oxy, A_causal_dxy = None, None
        if not args.disable_causal:
            X_train_chan_oxy = X_chan_oxy_orig[train_idx]
            X_train_chan_dxy = X_chan_dxy_orig[train_idx]
            y_train_raw = y_all[train_idx]

            train_adhd_mask = (y_train_raw == 0)
            train_hc_mask = (y_train_raw == 1)

            A_oxy_adhd = compute_causal_prior_from_channels(X_train_chan_oxy[train_adhd_mask], roi_mapping).to(device)
            A_oxy_hc = compute_causal_prior_from_channels(X_train_chan_oxy[train_hc_mask], roi_mapping).to(device)
            A_causal_oxy = torch.where((A_oxy_adhd != 0) | (A_oxy_hc != 0), 1.0, 0.0)

            A_dxy_adhd = compute_causal_prior_from_channels(X_train_chan_dxy[train_adhd_mask], roi_mapping).to(device)
            A_dxy_hc = compute_causal_prior_from_channels(X_train_chan_dxy[train_hc_mask], roi_mapping).to(device)
            A_causal_dxy = torch.where((A_dxy_adhd != 0) | (A_dxy_hc != 0), 1.0, 0.0)

        # --- 加载在原始数据上训练出的干净模型 ---
        model_path = os.path.join(args.model_dir, f'best_model_fold_{fold}.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到模型权重: {model_path}。请确保先运行未修改过的 main.py 训练出模型。")

        model = DualBranchRecurrentModel(
            embed_dim=args.hidden_dims, num_heads=args.head, depth=args.depth,
            k_memory=args.k_memory, num_classes=args.num_classes, drop=args.dropout,
            attn_drop=args.attn_drop, roi_mode=args.roi_mode, keep_ratio=args.keep_ratio,
            chunk_size=args.chunk_size, edl_mode=args.edl_mode
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # --- 准备测试集: 提取当前折的验证集 ---
        val_y = y_all[val_idx]

        # 1. 原始测试集
        val_oxy_orig = X_oxy_orig[val_idx]
        val_dxy_orig = X_dxy_orig[val_idx]

        # 2. 修正后的测试集 (HC保持不变，ADHD已被修正)
        val_oxy_corr = X_oxy_corr[val_idx]
        val_dxy_corr = X_dxy_corr[val_idx]

        # =============================================================
        # 测试 1: 在真实原始验证集上跑
        # =============================================================
        oxy_aug_orig, dxy_aug_orig, y_aug_orig = augment_data_odd_even(val_oxy_orig, val_dxy_orig, val_y)
        loader_orig = DataLoader(DualModalityDataset(oxy_aug_orig, dxy_aug_orig, y_aug_orig),
                                 batch_size=args.batch_size, shuffle=False)

        b_orig, u_orig, P_orig, preds_orig, labels_orig, probs_orig = evaluate_and_get_edl(model, loader_orig, device,
                                                                                           A_causal_oxy, A_causal_dxy)

        # 仅过滤出 ADHD 的样本进行收集与绘图
        adhd_idx_orig = (labels_orig == 0)
        orig_is_correct = (preds_orig == labels_orig).astype(int)

        results['original']['b'].extend(b_orig[adhd_idx_orig])
        results['original']['u'].extend(u_orig[adhd_idx_orig])
        results['original']['P'].extend(P_orig[adhd_idx_orig])
        results['original']['correct'].extend(orig_is_correct[adhd_idx_orig])

        acc = accuracy_score(labels_orig, preds_orig)
        pre = precision_score(labels_orig, preds_orig, zero_division=0)
        rec = recall_score(labels_orig, preds_orig, zero_division=0)
        f1 = f1_score(labels_orig, preds_orig, zero_division=0)
        try:
            auc = roc_auc_score(labels_orig, probs_orig)
        except:
            auc = 0.5
        try:
            nll = log_loss(labels_orig, probs_orig, labels=[0, 1])
        except:
            nll = 0.0
        ece = calculate_ece(P_orig, orig_is_correct)
        mean_b, mean_u, mean_P = np.mean(b_orig), np.mean(u_orig), np.mean(P_orig)

        results['original']['fold_metrics'].append([acc, f1, pre, rec, auc, nll, ece, mean_u, mean_b, mean_P])

        print(
            f"  -> [真实测试集 (Baseline)] ACC: {acc * 100:.2f} | F1: {f1 * 100:.2f} | PRE: {pre * 100:.2f} | REC: {rec * 100:.2f} | AUC: {auc * 100:.2f} | NLL: {nll * 100:.2f} | ECE: {ece * 100:.2f}")

        # =============================================================
        # 测试 2: 在干预/修正后的验证集上跑
        # =============================================================
        oxy_aug_corr, dxy_aug_corr, y_aug_corr = augment_data_odd_even(val_oxy_corr, val_dxy_corr, val_y)
        loader_corr = DataLoader(DualModalityDataset(oxy_aug_corr, dxy_aug_corr, y_aug_corr),
                                 batch_size=args.batch_size, shuffle=False)

        b_corr, u_corr, P_corr, preds_corr, labels_corr, probs_corr = evaluate_and_get_edl(model, loader_corr, device,
                                                                                           A_causal_oxy, A_causal_dxy)

        adhd_idx_corr = (labels_corr == 0)
        corr_is_correct = (preds_corr == labels_corr).astype(int)

        results['corrected']['b'].extend(b_corr[adhd_idx_corr])
        results['corrected']['u'].extend(u_corr[adhd_idx_corr])
        results['corrected']['P'].extend(P_corr[adhd_idx_corr])
        results['corrected']['correct'].extend(corr_is_correct[adhd_idx_corr])

        acc_c = accuracy_score(labels_corr, preds_corr)
        pre_c = precision_score(labels_corr, preds_corr, zero_division=0)
        rec_c = recall_score(labels_corr, preds_corr, zero_division=0)
        f1_c = f1_score(labels_corr, preds_corr, zero_division=0)
        try:
            auc_c = roc_auc_score(labels_corr, probs_corr)
        except:
            auc_c = 0.5
        try:
            nll_c = log_loss(labels_corr, probs_corr, labels=[0, 1])
        except:
            nll_c = 0.0
        ece_c = calculate_ece(P_corr, corr_is_correct)
        mean_b_c, mean_u_c, mean_P_c = np.mean(b_corr), np.mean(u_corr), np.mean(P_corr)

        results['corrected']['fold_metrics'].append(
            [acc_c, f1_c, pre_c, rec_c, auc_c, nll_c, ece_c, mean_u_c, mean_b_c, mean_P_c])

        print(
            f"  -> [修正测试集 (Intervened)] ACC: {acc_c * 100:.2f} | F1: {f1_c * 100:.2f} | PRE: {pre_c * 100:.2f} | REC: {rec_c * 100:.2f} | AUC: {auc_c * 100:.2f} | NLL: {nll_c * 100:.2f} | ECE: {ece_c * 100:.2f}")

    # ==========================================================
    # 输出五折平均指标
    # ==========================================================
    print("\n" + "=" * 80)
    print("========= 五折交叉验证 平均评估指标 ± 标准差 =========")
    for key in ['original', 'corrected']:
        avg_m = np.mean(results[key]['fold_metrics'], axis=0) * 100
        std_m = np.std(results[key]['fold_metrics'], axis=0) * 100
        print(f"[{key.upper()}]")
        print(
            f"  ACC: {avg_m[0]:.2f}±{std_m[0]:.2f} | F1: {avg_m[1]:.2f}±{std_m[1]:.2f} | PRE: {avg_m[2]:.2f}±{std_m[2]:.2f} | REC: {avg_m[3]:.2f}±{std_m[3]:.2f} | AUC: {avg_m[4]:.2f}±{std_m[4]:.2f} | NLL: {avg_m[5]:.2f}±{std_m[5]:.2f} | ECE: {avg_m[6]:.2f}±{std_m[6]:.2f}")
        print(
            f"  mean u: {avg_m[7]:.2f}±{std_m[7]:.2f} | mean b: {avg_m[8]:.2f}±{std_m[8]:.2f} | mean P: {avg_m[9]:.2f}±{std_m[9]:.2f}")
        print("-" * 80)

    # ==========================================================
    # 保存结果与绘制散点图
    # ==========================================================
    base_out_dir = os.path.join(args.model_dir, 'test_corrected_results')
    os.makedirs(base_out_dir, exist_ok=True)

    orig_b = np.array(results['original']['b'])
    orig_u = np.array(results['original']['u'])
    orig_P = np.array(results['original']['P'])
    orig_c = np.array(results['original']['correct'])

    corr_b = np.array(results['corrected']['b'])
    corr_u = np.array(results['corrected']['u'])
    corr_P = np.array(results['corrected']['P'])
    corr_c = np.array(results['corrected']['correct'])

    df = pd.DataFrame({
        'Original_b': orig_b, 'Original_u': orig_u, 'Original_P': orig_P, 'Original_Correct': orig_c,
        'Corrected_b': corr_b, 'Corrected_u': corr_u, 'Corrected_P': corr_P, 'Corrected_Correct': corr_c
    })
    csv_path = os.path.join(base_out_dir, 'correction_results.csv')
    df.to_csv(csv_path, index_label='Sample_Index')

    # 画散点图
    plt.figure(figsize=(9, 7))
    font_family = PLOT_FONT_CONFIG['family']

    plt.scatter(orig_b, orig_u, c='red', alpha=0.4, edgecolors='white', label='Original ADHD', s=60)
    plt.scatter(corr_b, corr_u, c='#1f77b4', alpha=0.8, edgecolors='white', label='Corrected ADHD (Phase & Amp)', s=80)

    for i in range(len(orig_b)):
        plt.arrow(orig_b[i], orig_u[i], corr_b[i] - orig_b[i], corr_u[i] - orig_u[i],
                  color='gray', alpha=0.25, width=0.0015, head_width=0.01)

    plt.xlabel('Belief (b)', fontdict={'family': font_family, 'size': 18})
    plt.ylabel('Uncertainty (u)', fontdict={'family': font_family, 'size': 18})
    plt.title('Causal Intervention: Corrected Phase & Amplitude', fontdict={'family': font_family, 'size': 20})
    plt.xticks(fontname=font_family, fontsize=14)
    plt.yticks(fontname=font_family, fontsize=14)
    plt.legend(prop={'family': font_family, 'size': 14})
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig(os.path.join(base_out_dir, 'u_b_scatter_corrected.png'), dpi=300)
    plt.close()

    print(f"\n✅ 修正测试完成！结果保存在 {base_out_dir}")


if __name__ == '__main__':
    main()