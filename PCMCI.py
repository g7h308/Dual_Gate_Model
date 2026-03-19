import numpy as np
import torch
from scipy.signal import decimate
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from statsmodels.stats.multitest import multipletests


def apply_global_signal_regression(fnirs_data):
    """
    执行全局信号回归 (GSR)
    输入 shape: (Batch, Channels, Time)
    输出 shape: (Batch, Channels, Time)
    """
    B, C, T = fnirs_data.shape
    cleaned_data = np.zeros_like(fnirs_data)

    for b in range(B):
        # 计算该样本下所有通道的平均信号作为全局混淆信号
        global_signal = np.mean(fnirs_data[b], axis=0)
        global_centered = global_signal - np.mean(global_signal)
        var_g = np.var(global_centered)

        for c in range(C):
            y = fnirs_data[b, c]
            y_centered = y - np.mean(y)

            if var_g == 0:
                cleaned_data[b, c] = y
            else:
                # 线性回归: 计算回归系数 beta
                beta = np.cov(global_centered, y_centered)[0, 1] / var_g
                # 减去被全局信号解释掉的部分，保留残差（即局部的特异性神经信号）
                cleaned_data[b, c] = y - beta * global_centered

    return cleaned_data


def compute_causal_prior_from_channels(fnirs_channel_data, roi_mapping, tau_max=10, pc_alpha=0.05,
                                       effect_size_threshold=0.1):
    B, C, T = fnirs_channel_data.shape
    num_rois = len(roi_mapping)

    # =========================================================
    # 步骤 1：全局信号回归 (去除头皮血流、呼吸等全局系统性噪声)
    # =========================================================
    cleaned_channel_data = apply_global_signal_regression(fnirs_channel_data)

    # =========================================================
    # 步骤 2：空间映射 (转换为 ROI 级别数据)
    # =========================================================
    roi_data = np.zeros((B, num_rois, T))
    for i, channels in enumerate(roi_mapping):
        roi_data[:, i, :] = np.mean(cleaned_channel_data[:, channels, :], axis=1)

    # =========================================================
    # 步骤 3：抗混叠降采样 (Decimate)
    # 自动应用低通滤波后再抽头，防止高频噪声混叠到低频
    # =========================================================
    downsample_factor = 10
    roi_data_downsampled = decimate(roi_data, q=downsample_factor, ftype='iir', axis=-1)

    # =========================================================
    # 步骤 4：一阶差分 (打破残存的极高自相关性)
    # 计算变化率，此时时间维度会减 1
    # =========================================================
    roi_data_diff = np.diff(roi_data_downsampled, axis=-1)

    # =========================================================
    # 步骤 5：Tigramite PCMCI 因果推断
    # =========================================================
    data_for_tigramite = np.transpose(roi_data_diff, (0, 2, 1))
    var_names = [f'ROI_{i}' for i in range(num_rois)]

    dataframe = pp.DataFrame(
        data_for_tigramite,
        var_names=var_names,
        analysis_mode='multiple'
    )

    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

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
    # 步骤 6：FDR 校正与效应量双重过滤
    # =========================================================
    p_flat = p_matrix.flatten()
    valid_mask = ~np.isnan(p_flat)
    q_flat = np.ones_like(p_flat)

    _, q_values, _, _ = multipletests(p_flat[valid_mask], alpha=pc_alpha, method='fdr_bh')
    q_flat[valid_mask] = q_values
    q_matrix = q_flat.reshape(p_matrix.shape)

    clean_causal_matrix = np.where(
        (q_matrix < pc_alpha) & (np.abs(val_matrix) > effect_size_threshold),
        val_matrix,
        0.0
    )

    clean_causal_matrix[:, :, 0] = 0.0  # 强制去除瞬时效应 (tau=0)

    return torch.tensor(clean_causal_matrix, dtype=torch.float32)