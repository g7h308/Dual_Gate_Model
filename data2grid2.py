# import numpy as np
#
#
# def convert_channel_to_grid(channel_data):
#     """
#     将形状为 (样本数, 通道数, 时间步) 的数据
#     映射为形状为 (样本数, 时间步, 高, 宽) 的网格数据
#     """
#     N_samples, C_channels, T_time = channel_data.shape
#
#     # 初始化一个全0的网格矩阵 (样本数, 1600, 5, 9)
#     grid_data = np.zeros((N_samples, T_time, 5, 9))
#
#     # 按照 data2grid.py 中的传感器坐标映射表
#     coords_map = {
#         0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
#         4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
#         9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
#         13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
#         18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
#     }
#
#     # 批量将通道数据填入对应的二维网格坐标系中
#     for ch, (r, c) in coords_map.items():
#         # grid_data[:, :, r, c] 对应所有样本、所有时间点的那个像素点
#         grid_data[:, :, r, c] = channel_data[:, ch, :]
#
#     return grid_data
#
#
# def main():
#     print("正在加载修正后的双步法 ADHD 通道数据...")
#     # 加载上一步生成的 (47, 22, 1600) 数据
#     adhd_oxy_chan = np.load('Corrected_ADHD_oxy_TwoSteps.npy')
#     adhd_dxy_chan = np.load('Corrected_ADHD_dxy_TwoSteps.npy')
#
#     print("正在将其映射为 5x9 空间网格特征...")
#     adhd_oxy_grid = convert_channel_to_grid(adhd_oxy_chan)
#     adhd_dxy_grid = convert_channel_to_grid(adhd_dxy_chan)
#
#     # 保存为可以喂给深度学习模型的三维特征图格式
#     out_oxy = 'ADHD_grid_oxy_corrected.npy'
#     out_dxy = 'ADHD_grid_dxy_corrected.npy'
#
#     np.save(out_oxy, adhd_oxy_grid)
#     np.save(out_dxy, adhd_dxy_grid)
#
#     print("\n==========================================")
#     print("转换完成！")
#     print(f"修正后的 ADHD Oxy 网格已保存: {out_oxy} (形状: {adhd_oxy_grid.shape})")
#     print(f"修正后的 ADHD Dxy 网格已保存: {out_dxy} (形状: {adhd_dxy_grid.shape})")
#     print("==========================================")
#     print("\n[下一步提示]：")
#     print("你可以直接把这两个文件移动到你的 data/VFT/ 目录下，")
#     print("并重命名为 ADHD_grid_oxy.npy 和 ADHD_grid_dxy.npy (覆盖原文件)。")
#
#
# if __name__ == '__main__':
#     main()

