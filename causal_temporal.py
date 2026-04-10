import os
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

# 导入必要模块，增加 PCMCI 和通道数据加载
from dataloader.VFTDataLoader import load_raw_data, augment_data_odd_even, DualModalityDataset, \
    load_excel_channel_data_dual
from models.DualBranchModel import DualBranchRecurrentModel
from PCMCI import compute_causal_prior_from_channels
from tool import set_seed


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    parser.add_argument('--model_dir', type=str, default='./checkpoints/20260410_155626_causal_temporal')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--hidden_dims', type=int, default=64)
    parser.add_argument('--head', type=int, default=2)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--k_memory', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--attn_drop', type=float, default=0.4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--roi_mode', type=str, default='original')
    parser.add_argument('--keep_ratio', type=float, default=1.0)
    parser.add_argument('--chunk_size', type=int, default=10)
    parser.add_argument('--edl_mode', type=bool, default=True)
    return parser.parse_args()


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


def get_hc_mean_features(model, hc_loader, device, A_causal_oxy, A_causal_dxy):
    """提取 HC 样本特征时，也必须带上因果掩码"""
    model.eval()
    all_hbo2, all_hbr = [], []
    with torch.no_grad():
        for oxy, dxy, labels in hc_loader:
            oxy, dxy = oxy.to(device), dxy.to(device)
            # 加入空间掩码以保证特征提取结构一致
            hbo2_feat, hbr_feat = model(oxy, dxy, A_causal_oxy=A_causal_oxy, A_causal_dxy=A_causal_dxy,
                                        return_step_features=True)
            all_hbo2.append(hbo2_feat.cpu())
            all_hbr.append(hbr_feat.cpu())

    all_hbo2 = torch.cat(all_hbo2, dim=0)
    all_hbr = torch.cat(all_hbr, dim=0)
    mean_hbo2 = torch.mean(all_hbo2, dim=0)
    mean_hbr = torch.mean(all_hbr, dim=0)
    return mean_hbo2, mean_hbr


# 新增 A_causal_oxy 和 A_causal_dxy 参数
def evaluate_temporal_ablation(model, loader, device, A_causal_oxy, A_causal_dxy, intervene_dict=None):
    model.eval()
    all_b, all_u, all_P = [], [], []
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for oxy, dxy, labels in loader:
            oxy, dxy, labels = oxy.to(device), dxy.to(device), labels.to(device)
            # 正确传入训练集的空间因果先验图
            outputs = model(oxy, dxy, A_causal_oxy=A_causal_oxy, A_causal_dxy=A_causal_dxy,
                            intervene_dict=intervene_dict)

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
    X_oxy_all, X_dxy_all, y_all = load_raw_data(args)

    # -------------------------------------------------------------
    # 新增：加载通道数据以供后续 PCMCI 计算
    target_len = X_oxy_all.shape[1] + 1
    X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)
    X_chan_oxy_all = np.delete(X_chan_oxy_all, 0, axis=2)
    X_chan_dxy_all = np.delete(X_chan_dxy_all, 0, axis=2)

    if args.roi_mode == 'grid_1x3':
        roi_mapping = [[0], [1, 2], [3], [4, 5], [6], [7, 8], [9], [10, 11], [12], [13, 14], [15], [16, 17], [18],
                       [19, 20], [21]]
    else:
        roi_mapping = [[0, 4, 5], [1, 2], [3, 7, 8], [9, 13, 14, 18], [6, 10, 11, 15, 19, 20], [12, 16, 17, 21]]
    # -------------------------------------------------------------

    augmented_time_steps = X_oxy_all.shape[1] // 2
    num_steps = augmented_time_steps // args.chunk_size
    p1 = int(num_steps * (200 / 800))
    p2 = int(num_steps * (500 / 800))

    segments = {
        "Seg_0_200": range(0, p1),
        "Seg_200_500": range(p1, p2),
        "Seg_500_800": range(p2, num_steps)
    }

    results = {'Baseline': {'b': [], 'u': [], 'P': [], 'fold_metrics': [], 'is_correct': []}}
    for seg_name in segments.keys():
        results[seg_name] = {'b': [], 'u': [], 'P': [], 'fold_metrics': [], 'is_correct': []}

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    print("\n" + "=" * 60)
    print("开始进行 5 折时间因果消融评估 (带空间因果掩码)...")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_oxy_all)):
        print(f"\n{'=' * 20} Processing Fold {fold + 1} {'=' * 20}")

        # -------------------------------------------------------------
        # 新增：严格使用训练集实时计算空间因果图
        train_hc_mask_raw = (y_all[train_idx] == 1)
        train_adhd_mask_raw = (y_all[train_idx] == 0)

        A_oxy_adhd = compute_causal_prior_from_channels(X_chan_oxy_all[train_idx][train_adhd_mask_raw], roi_mapping).to(
            device)
        A_oxy_hc = compute_causal_prior_from_channels(X_chan_oxy_all[train_idx][train_hc_mask_raw], roi_mapping).to(
            device)
        A_causal_oxy = torch.where((A_oxy_adhd != 0) | (A_oxy_hc != 0), 1.0, 0.0)

        A_dxy_adhd = compute_causal_prior_from_channels(X_chan_dxy_all[train_idx][train_adhd_mask_raw], roi_mapping).to(
            device)
        A_dxy_hc = compute_causal_prior_from_channels(X_chan_dxy_all[train_idx][train_hc_mask_raw], roi_mapping).to(
            device)
        A_causal_dxy = torch.where((A_dxy_adhd != 0) | (A_dxy_hc != 0), 1.0, 0.0)
        # -------------------------------------------------------------

        model_path = os.path.join(args.model_dir, f'best_model_fold_{fold}.pt')
        model = DualBranchRecurrentModel(
            embed_dim=args.hidden_dims, num_heads=args.head, depth=args.depth,
            k_memory=args.k_memory, num_classes=args.num_classes, drop=args.dropout,
            attn_drop=args.attn_drop, roi_mode=args.roi_mode, keep_ratio=args.keep_ratio,
            chunk_size=args.chunk_size, edl_mode=args.edl_mode
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        train_hc_oxy = X_oxy_all[train_idx][train_hc_mask_raw]
        train_hc_dxy = X_dxy_all[train_idx][train_hc_mask_raw]
        train_hc_y = y_all[train_idx][train_hc_mask_raw]
        train_hc_oxy_aug, train_hc_dxy_aug, train_hc_y_aug = augment_data_odd_even(train_hc_oxy, train_hc_dxy,
                                                                                   train_hc_y)
        train_hc_loader = DataLoader(DualModalityDataset(train_hc_oxy_aug, train_hc_dxy_aug, train_hc_y_aug),
                                     batch_size=args.batch_size, shuffle=False)

        # 获取健康模板也传入因果图
        mean_hbo2, mean_hbr = get_hc_mean_features(model, train_hc_loader, device, A_causal_oxy, A_causal_dxy)

        val_oxy_aug, val_dxy_aug, val_y_aug = augment_data_odd_even(X_oxy_all[val_idx], X_dxy_all[val_idx],
                                                                    y_all[val_idx])
        val_loader = DataLoader(DualModalityDataset(val_oxy_aug, val_dxy_aug, val_y_aug), batch_size=args.batch_size,
                                shuffle=False)

        def run_eval_and_store(key, intervene_dict, log_prefix="Baseline"):
            b, u, P, preds, labels, probs = evaluate_temporal_ablation(model, val_loader, device, A_causal_oxy,
                                                                       A_causal_dxy, intervene_dict)

            acc = accuracy_score(labels, preds)
            pre = precision_score(labels, preds, zero_division=0)
            rec = recall_score(labels, preds, zero_division=0)
            f1 = f1_score(labels, preds, zero_division=0)
            try:
                auc = roc_auc_score(labels, probs)
            except:
                auc = 0.5
            try:
                nll = log_loss(labels, probs, labels=[0, 1])
            except:
                nll = 0.0

            is_correct = (preds == labels).astype(int)
            ece = calculate_ece(P, is_correct)

            adhd_idx = (labels == 0)
            mean_b = np.mean(b[adhd_idx])
            mean_u = np.mean(u[adhd_idx])
            mean_P = np.mean(P[adhd_idx])

            results[key]['fold_metrics'].append([acc, f1, pre, rec, auc, nll, ece, mean_u, mean_b, mean_P])
            results[key]['b'].extend(b[adhd_idx])
            results[key]['u'].extend(u[adhd_idx])
            results[key]['P'].extend(P[adhd_idx])
            results[key]['is_correct'].extend(is_correct[adhd_idx])

            print(
                f"  -> [{log_prefix}] ACC: {acc * 100:.2f} | F1: {f1 * 100:.2f} | PRE: {pre * 100:.2f} | REC: {rec * 100:.2f} | AUC: {auc * 100:.2f} | NLL: {nll * 100:.2f} | ECE: {ece * 100:.2f}")
            print(
                f"                       mean u: {mean_u * 100:.2f} | mean b: {mean_b * 100:.2f} | mean P: {mean_P * 100:.2f}")

        # 1. 评估 Baseline
        run_eval_and_store('Baseline', intervene_dict=None, log_prefix="Unmasked Baseline")

        # 2. 评估各个时间消融段
        for seg_name, step_range in segments.items():
            current_intervene = {step: {'hbo2': mean_hbo2[step], 'hbr': mean_hbr[step]} for step in step_range}
            run_eval_and_store(seg_name, current_intervene, log_prefix=f"Masked {seg_name}")

    print("\n" + "=" * 80)
    print("========= 五折交叉验证 平均评估指标 ± 标准差 =========")
    print("=" * 80)

    for key in results.keys():
        avg_m = np.mean(results[key]['fold_metrics'], axis=0) * 100
        std_m = np.std(results[key]['fold_metrics'], axis=0) * 100
        print(f"[{key.upper()}]")
        print(
            f"  ACC: {avg_m[0]:.2f}±{std_m[0]:.2f} | F1: {avg_m[1]:.2f}±{std_m[1]:.2f} | PRE: {avg_m[2]:.2f}±{std_m[2]:.2f} | REC: {avg_m[3]:.2f}±{std_m[3]:.2f} | AUC: {avg_m[4]:.2f}±{std_m[4]:.2f} | NLL: {avg_m[5]:.2f}±{std_m[5]:.2f} | ECE: {avg_m[6]:.2f}±{std_m[6]:.2f}")
        print(
            f"  mean u: {avg_m[7]:.2f}±{std_m[7]:.2f} | mean b: {avg_m[8]:.2f}±{std_m[8]:.2f} | mean P: {avg_m[9]:.2f}±{std_m[9]:.2f}")
        print("-" * 80)

    out_dir = os.path.join(args.model_dir, 'temporal_causal_results')
    os.makedirs(out_dir, exist_ok=True)
    excel_save_path = os.path.join(out_dir, 'temporal_ablation_results.xlsx')
    with pd.ExcelWriter(excel_save_path) as writer:
        for key in results.keys():
            df = pd.DataFrame({
                'b': results[key]['b'],
                'u': results[key]['u'],
                'P': results[key]['P'],
                'is_correct': results[key]['is_correct']  # <--- 新增这一列
            })
            df.to_excel(writer, sheet_name=key, index=False)

    print(f"\n✅ 完成！结果表已保存至: {excel_save_path}")


if __name__ == "__main__":
    main()