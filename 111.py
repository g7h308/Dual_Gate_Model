# import numpy as np
# import pandas as pd
# import argparse
# import scipy.stats as st
# from scipy.signal import hilbert
# import os
#
# # 导入你项目中的数据加载函数
# from dataloader.VFTDataLoader import load_raw_data, load_excel_channel_data_dual
#
#
# def get_args():
#     parser = argparse.ArgumentParser()
#     # 默认路径与 causal.py 保持一致
#     parser.add_argument('--data_path', type=str, default='./data/VFT')
#     return parser.parse_args()
#
#
# def calc_phase_diff(sig1, sig2):
#     """计算两个信号之间的平均相位差"""
#     analytic_signal1 = hilbert(sig1)
#     analytic_signal2 = hilbert(sig2)
#     phase1 = np.unwrap(np.angle(analytic_signal1))
#     phase2 = np.unwrap(np.angle(analytic_signal2))
#     return np.mean(phase1 - phase2)
#
#
# def calc_amp_ratio(sig1, sig2):
#     """计算振幅比 (使用标准差近似振幅)"""
#     std2 = np.std(sig2)
#     if std2 == 0:
#         return 0
#     return np.std(sig1) / std2
#
#
# def main():
#     args = get_args()
#
#     # ==========================================
#     # 1. 真实数据加载逻辑
#     # ==========================================
#     print(f"正在从 {args.data_path} 加载数据标签...")
#     _, _, y_all = load_raw_data(args)
#
#     target_len = 1600
#     print(f"正在从 Excel 提取 {target_len} 长度的原始双模态通道数据...")
#     X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)
#
#     # 标签分离数据
#     adhd_mask = (y_all == 0)
#     hc_mask = (y_all == 1)
#
#     adhd_oxy = X_chan_oxy_all[adhd_mask]
#     adhd_dxy = X_chan_dxy_all[adhd_mask]
#
#     hc_oxy = X_chan_oxy_all[hc_mask]
#     hc_dxy = X_chan_dxy_all[hc_mask]
#
#     num_adhd_samples = adhd_oxy.shape[0]
#     num_hc_samples = hc_oxy.shape[0]
#     num_channels = X_chan_oxy_all.shape[1]
#
#     print(f"数据加载完毕！ADHD 样本数: {num_adhd_samples}, HC 样本数: {num_hc_samples}, 通道数: {num_channels}")
#
#     # ==========================================
#     # 2. 定义切片范围与 DataFrame 列名
#     # ==========================================
#     segments = [(0, 400), (400, 1000), (1000, 1600)]
#     seg_names = ['0-400', '401-1000', '1001-1600']
#
#     columns = []
#     for ch in range(num_channels):
#         for seg_name in seg_names:
#             columns.append(f'Ch{ch}_{seg_name}_PhaseDiff')
#             columns.append(f'Ch{ch}_{seg_name}_AmpRatio')
#
#     # ==========================================
#     # 3. 处理 HC 样本 (计算平均值与 95% 置信区间)
#     # ==========================================
#     print("正在计算 HC 样本指标与置信区间...")
#     hc_all_samples_metrics = []
#
#     for i in range(num_hc_samples):
#         sample_metrics = []
#         for ch in range(num_channels):
#             for start, end in segments:
#                 sig_oxy = hc_oxy[i, ch, start:end]
#                 sig_dxy = hc_dxy[i, ch, start:end]
#
#                 phase_diff = calc_phase_diff(sig_oxy, sig_dxy)
#                 amp_ratio = calc_amp_ratio(sig_oxy, sig_dxy)
#
#                 sample_metrics.extend([phase_diff, amp_ratio])
#         hc_all_samples_metrics.append(sample_metrics)
#
#     hc_all_samples_metrics = np.array(hc_all_samples_metrics)  # 形状: (47, 132)
#
#     # 计算均值
#     hc_mean = np.mean(hc_all_samples_metrics, axis=0)
#
#     # 计算 95% 置信区间 (使用 scipy.stats.t.interval)
#     # std error of mean (SEM)
#     hc_sem = st.sem(hc_all_samples_metrics, axis=0)
#     hc_ci_lower, hc_ci_upper = st.t.interval(confidence=0.95,
#                                              df=num_hc_samples - 1,
#                                              loc=hc_mean,
#                                              scale=hc_sem)
#
#     # 组装 HC DataFrame，包含3行：均值、下界、上界
#     df_hc = pd.DataFrame([hc_mean, hc_ci_lower, hc_ci_upper], columns=columns)
#     df_hc.insert(0, 'Metric_Type', ['HC_Average', 'HC_CI_Lower_95', 'HC_CI_Upper_95'])
#
#     # ==========================================
#     # 4. 处理 ADHD 样本
#     # ==========================================
#     print("正在计算 ADHD 样本指标...")
#     adhd_all_samples_metrics = []
#
#     for i in range(num_adhd_samples):
#         sample_metrics = []
#         for ch in range(num_channels):
#             for start, end in segments:
#                 sig_oxy = adhd_oxy[i, ch, start:end]
#                 sig_dxy = adhd_dxy[i, ch, start:end]
#
#                 phase_diff = calc_phase_diff(sig_oxy, sig_dxy)
#                 amp_ratio = calc_amp_ratio(sig_oxy, sig_dxy)
#
#                 sample_metrics.extend([phase_diff, amp_ratio])
#         adhd_all_samples_metrics.append(sample_metrics)
#
#     df_adhd = pd.DataFrame(adhd_all_samples_metrics, columns=columns)
#     df_adhd.insert(0, 'Sample_ID', [f'ADHD_{i + 1}' for i in range(num_adhd_samples)])
#
#     # ==========================================
#     # 5. 结果保存
#     # ==========================================
#     output_path = "Segmented_Metrics_Results_with_CI.xlsx"
#     with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
#         df_hc.to_excel(writer, sheet_name='HC', index=False)
#         df_adhd.to_excel(writer, sheet_name='ADHD', index=False)
#
#     print(f"\n==========================================")
#     print(f"处理成功！结果已保存至: {output_path}")
#     print(f" [HC sheet 形状]:   {df_hc.shape} -> (3行统计量[均值/下界/上界] + 标识列 + 132个指标)")
#     print(f" [ADHD sheet 形状]: {df_adhd.shape} -> ({num_adhd_samples}个样本 + 标识列 + 132个指标)")
#     print(f"==========================================")
#
#
# if __name__ == '__main__':
#     main()


# import pandas as pd
# import os
#
#
# def main():
#     file_path = "Segmented_Metrics_Results_with_CI.xlsx"
#
#     if not os.path.exists(file_path):
#         print(f"错误: 找不到文件 {file_path}，请先运行上一步的生成脚本。")
#         return
#
#     print(f"正在读取 {file_path} ...")
#     # 读取 HC 和 ADHD 的数据
#     df_hc = pd.read_excel(file_path, sheet_name='HC')
#     df_adhd = pd.read_excel(file_path, sheet_name='ADHD')
#
#     # 从 HC 表中提取置信区间的下界和上界
#     lower_bounds = df_hc[df_hc['Metric_Type'] == 'HC_CI_Lower_95'].iloc[0]
#     upper_bounds = df_hc[df_hc['Metric_Type'] == 'HC_CI_Upper_95'].iloc[0]
#
#     # 定义通道数和时间段（与上一步保持一致）
#     num_channels = 22
#     seg_names = ['0-400', '401-1000', '1001-1600']
#
#     results = []
#
#     print("正在对比 ADHD 样本与 HC 置信区间...")
#     # 遍历 ADHD 的每一行（每一个样本）
#     for idx, row in df_adhd.iterrows():
#         sample_id = row['Sample_ID']
#
#         # 遍历每一个通道和每一个时间段
#         for ch in range(num_channels):
#             for seg in seg_names:
#                 col_pd = f'Ch{ch}_{seg}_PhaseDiff'
#                 col_ar = f'Ch{ch}_{seg}_AmpRatio'
#
#                 # 获取该 ADHD 样本的实际值
#                 val_pd = row[col_pd]
#                 val_ar = row[col_ar]
#
#                 # 获取对应 HC 组的置信区间 [下界, 上界]
#                 low_pd, up_pd = lower_bounds[col_pd], upper_bounds[col_pd]
#                 low_ar, up_ar = lower_bounds[col_ar], upper_bounds[col_ar]
#
#                 # 判断是否在置信区间之外
#                 pd_out = (val_pd < low_pd) or (val_pd > up_pd)
#                 ar_out = (val_ar < low_ar) or (val_ar > up_ar)
#
#                 # 记录结果
#                 status = None
#                 if pd_out and ar_out:
#                     status = "二者都不在"
#                 elif pd_out:
#                     status = "相位差不在"
#                 elif ar_out:
#                     status = "振幅比不在"
#
#                 if status:
#                     results.append({
#                         "样本_通道_时间段": f"{sample_id}_Ch{ch}_{seg}",
#                         "不在置信区间内的指标": status
#                     })
#
#     # 将结果转换为 DataFrame
#     df_out = pd.DataFrame(results)
#
#     print(f"分析完成！共发现 {len(df_out)} 条异常记录。正在保存到 Excel 中...")
#
#     # 使用 openpyxl 引擎以追加模式 (mode='a') 写入原文件
#     # if_sheet_exists='replace' 保证如果多次运行，会自动覆盖这个新Sheet而不是报错
#     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#         df_out.to_excel(writer, sheet_name='异常指标分析', index=False)
#
#     print(f"保存成功！请打开 {file_path} 查看名为 '异常指标分析' 的新 Sheet。")
#
#
# if __name__ == '__main__':
#     main()

import numpy as np
import pandas as pd
import argparse
import os
from scipy.signal import hilbert

# 导入你项目中的数据加载函数
from dataloader.VFTDataLoader import load_raw_data, load_excel_channel_data_dual


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/VFT')
    return parser.parse_args()


def main():
    args = get_args()
    file_path = "Segmented_Metrics_Results_with_CI.xlsx"

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # ==========================================
    # 1. 加载异常记录与 HC 统计值
    # ==========================================
    print(f"正在读取异常记录和 HC 均值...")
    df_anomalies = pd.read_excel(file_path, sheet_name='异常指标分析')
    df_hc = pd.read_excel(file_path, sheet_name='HC')
    hc_avg = df_hc[df_hc['Metric_Type'] == 'HC_Average'].iloc[0]

    # ==========================================
    # 2. 加载原始时间序列数据
    # ==========================================
    print(f"正在加载原始双模态数据...")
    _, _, y_all = load_raw_data(args)
    target_len = 1600
    X_chan_oxy_all, X_chan_dxy_all = load_excel_channel_data_dual(args.data_path, target_len=target_len)

    adhd_mask = (y_all == 0)
    adhd_oxy = X_chan_oxy_all[adhd_mask].copy()
    adhd_dxy = X_chan_dxy_all[adhd_mask].copy()

    # ==========================================
    # 3. 两步法修正：代数修正振幅比 + 希尔伯特修正相位差
    # ==========================================
    segments_map = {
        '0-400': (0, 400),
        '401-1000': (400, 1000),
        '1001-1600': (1000, 1600)
    }

    print(f"开始执行两步法修正，共处理 {len(df_anomalies)} 个异常片段...")

    for idx, row in df_anomalies.iterrows():
        identifier = row['样本_通道_时间段']
        parts = identifier.split('_')
        sample_idx = int(parts[1]) - 1
        ch_idx = int(parts[2].replace('Ch', ''))
        seg_str = parts[3]

        start, end = segments_map[seg_str]

        # 提取 HC 组健康的振幅比和相位差
        col_ar = f'Ch{ch_idx}_{seg_str}_AmpRatio'
        col_pd = f'Ch{ch_idx}_{seg_str}_PhaseDiff'
        R_HC = hc_avg[col_ar]
        Phi_HC = hc_avg[col_pd]  # 弧度制相位差

        # 提取当前异常片段
        oxy_seg = adhd_oxy[sample_idx, ch_idx, start:end]
        dxy_seg = adhd_dxy[sample_idx, ch_idx, start:end]

        # -----------------------------------------------------
        # 步骤 1：严格使用代数公式，强制对齐振幅比 (此时相位差为0)
        # -----------------------------------------------------
        hbt_seg = oxy_seg + dxy_seg
        oxy_tmp = hbt_seg * (R_HC / (1.0 + R_HC))
        dxy_tmp = hbt_seg / (1.0 + R_HC)

        # -----------------------------------------------------
        # 步骤 2：使用希尔伯特变换提取解析信号，只做相位旋转
        # -----------------------------------------------------
        # 获取解析信号
        z_oxy = hilbert(oxy_tmp)
        z_dxy = hilbert(dxy_tmp)

        # 对称注入相位差，使得 Phase(oxy) - Phase(dxy) = Phi_HC
        z_oxy_shifted = z_oxy * np.exp(1j * (Phi_HC / 2.0))
        z_dxy_shifted = z_dxy * np.exp(-1j * (Phi_HC / 2.0))

        # 取实部还原为时间序列物理信号
        new_oxy = np.real(z_oxy_shifted)
        new_dxy = np.real(z_dxy_shifted)

        # 将修正后的数据写回
        adhd_oxy[sample_idx, ch_idx, start:end] = new_oxy
        adhd_dxy[sample_idx, ch_idx, start:end] = new_dxy

    # ==========================================
    # 4. 存储修正后的数据
    # ==========================================
    out_oxy_file = "Corrected_ADHD_oxy_TwoSteps.npy"
    out_dxy_file = "Corrected_ADHD_dxy_TwoSteps.npy"

    np.save(out_oxy_file, adhd_oxy)
    np.save(out_dxy_file, adhd_dxy)

    print("\n==========================================")
    print("两步法修正完成！")
    print(f"Oxy 保存至: {out_oxy_file}")
    print(f"Dxy 保存至: {out_dxy_file}")
    print("==========================================")


if __name__ == '__main__':
    main()