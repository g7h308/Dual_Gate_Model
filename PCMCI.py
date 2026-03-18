import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import torch

# 【新增】：导入老牌统计库进行极其稳定可靠的 FDR 校正
from statsmodels.stats.multitest import multipletests


# --- 修改后的因果先验计算函数 ---
def compute_causal_prior_from_channels(fnirs_channel_data, roi_mapping, tau_max=10, pc_alpha=0.05):
    B, C, T = fnirs_channel_data.shape
    num_rois = len(roi_mapping)

    roi_data = np.zeros((B, num_rois, T))
    for i, channels in enumerate(roi_mapping):
        roi_data[:, i, :] = np.mean(fnirs_channel_data[:, channels, :], axis=1)

    # =========================================================
    # 【修改 1：时间维度下采样 10 倍】
    # 切片操作 [::10]，每 10 帧取 1 帧，极大拉开时间跨度，彰显真实因果延迟
    # =========================================================
    roi_data = roi_data[:, :, ::10]

    data_for_tigramite = np.transpose(roi_data, (0, 2, 1))

    var_names = [f'ROI_{i}' for i in range(num_rois)]

    dataframe = pp.DataFrame(
        data_for_tigramite,
        var_names=var_names,
        analysis_mode='multiple'
    )

    cond_ind_test = ParCorr(significance='analytic')

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    # 限制搜索深度，防止计算爆炸
    results = pcmci.run_pcmci(
        tau_max=tau_max,
        pc_alpha=pc_alpha,
        max_conds_dim=2,
        max_conds_px=2,
        max_conds_py=2
    )

    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']

    # =========================================================
    # 【修改 2：FDR (Benjamini-Hochberg) 校正】
    # =========================================================
    # a. 将多维的 p_matrix 展平以便批量校正
    p_flat = p_matrix.flatten()

    # b. 排除 NaN 值（比如对角线的 tau=0 无法计算因果）
    valid_mask = ~np.isnan(p_flat)
    q_flat = np.ones_like(p_flat)  # 默认 q 值设为 1.0 (绝对不显著)

    # c. 调用 statsmodels 的 multipletests 进行 FDR_bh 校正
    # q_values 就是经过严苛惩罚后新的 "p值"
    _, q_values, _, _ = multipletests(p_flat[valid_mask], alpha=pc_alpha, method='fdr_bh')
    q_flat[valid_mask] = q_values

    # d. 还原回 6x6x11 的矩阵形状
    q_matrix = q_flat.reshape(p_matrix.shape)

    # =========================================================
    # 最终过滤：必须满足 FDR 校正后的 q值 < pc_alpha
    # （可选：你依然可以像之前那样在这个条件里加上 & (np.abs(val_matrix) > 0.1) 来进一步过滤极弱的效应量）
    # =========================================================
    clean_causal_matrix = np.where(
        q_matrix < pc_alpha,
        val_matrix,
        0.0
    )

    clean_causal_matrix[:, :, 0] = 0.0  # 去除瞬时效应

    return torch.tensor(clean_causal_matrix, dtype=torch.float32)