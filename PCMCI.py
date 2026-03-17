import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import torch


# --- 新增因果先验计算函数 ---
def compute_causal_prior_from_channels(fnirs_channel_data, roi_mapping, tau_max=10, pc_alpha=0.05):
    B, C, T = fnirs_channel_data.shape
    num_rois = len(roi_mapping)

    roi_data = np.zeros((B, num_rois, T))
    for i, channels in enumerate(roi_mapping):
        roi_data[:, i, :] = np.mean(fnirs_channel_data[:, channels, :], axis=1)

    data_for_tigramite = np.transpose(roi_data, (0, 2, 1))

    var_names = [f'ROI_{i}' for i in range(num_rois)]

    # 2. 必须显式增加 analysis_mode='multiple' 参数！
    dataframe = pp.DataFrame(
        data_for_tigramite,
        var_names=var_names,
        analysis_mode='multiple'  # 告诉算法这是多个独立受试者的数据
    )

    cond_ind_test = ParCorr(significance='analytic')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)

    # run_pcmci 里面只留核心参数
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha, max_conds_dim=2, max_conds_px=2, max_conds_py=2)

    clean_causal_matrix = np.where(results['p_matrix'] < pc_alpha, results['val_matrix'], 0.0)
    clean_causal_matrix[:, :, 0] = 0.0  # 去除瞬时效应

    return torch.tensor(clean_causal_matrix, dtype=torch.float32)