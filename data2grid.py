import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata


# ==========================================
# 1. 核心处理函数 (包含标准化、插值、角落平均)
# ==========================================
def process_single_file(file_path):
    """
    处理单个 Excel 文件，返回 (N, 5, 9) 的矩阵数组
    """
    # 1. 读取
    try:
        df = pd.read_excel(file_path, header=None)
    except Exception as e:
        print(f"  [失败] 无法读取文件 {os.path.basename(file_path)}: {e}")
        return None

    data = df.values

    # 校验列数
    if data.shape[1] < 22:
        print(f"  [跳过] 列数不足 ({data.shape[1]} < 22): {os.path.basename(file_path)}")
        return None

    sensor_data = data[:, :22]

    # 2. 标准化 (Standardization)
    # 注意：这里是对【当前文件】进行独立标准化。
    # 如果希望所有文件统一标准，需要先合并所有数据再标准化(消耗内存大)，
    # 或者保存scaler参数。通常独立标准化也是可行的。
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(sensor_data)

    # 3. 坐标映射与网格构建
    coords_map = {
        0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
        4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
        9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
        13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
        18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
    }
    points = np.array([coords_map[i] for i in range(22)])
    grid_x, grid_y = np.mgrid[0:5:1, 0:9:1]

    processed_matrices = []

    # 4. 逐行插值处理
    for row_values in data_norm:
        # A. 三次样条插值 (Cubic)
        grid_z = griddata(points, row_values, (grid_x, grid_y), method='cubic')

        # B. 填充 NaN (Nearest)
        if np.isnan(grid_z).any():
            grid_z_nearest = griddata(points, row_values, (grid_x, grid_y), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

        # C. 修正四个角落 (取平均值)
        # 左上 (0,0)
        grid_z[0, 0] = (grid_z[0, 1] + grid_z[1, 0]) / 2.0
        # 右上 (0,8)
        grid_z[0, 8] = (grid_z[0, 7] + grid_z[1, 8]) / 2.0
        # 左下 (4,0)
        grid_z[4, 0] = (grid_z[3, 0] + grid_z[4, 1]) / 2.0
        # 右下 (4,8)
        grid_z[4, 8] = (grid_z[3, 8] + grid_z[4, 7]) / 2.0

        processed_matrices.append(grid_z)

    return np.array(processed_matrices)


# ==========================================
# 2. 批量处理主程序
# ==========================================
def batch_process_folder(input_folder, output_file):
    print(f"--- 开始处理文件夹: {input_folder} ---")

    # 获取所有 .xlsx 和 .xls 文件
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx")) + \
                glob.glob(os.path.join(input_folder, "*.xls"))

    if not all_files:
        print("未找到Excel文件！请检查路径。")
        return

    all_data_list = []
    file_record = []  # 记录数据来源，方便后续追溯

    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        print(f"[{i + 1}/{len(all_files)}] 处理中: {file_name} ...", end="")

        matrix_data = process_single_file(file_path)

        if matrix_data is not None:
            all_data_list.append(matrix_data)
            # 记录这个文件包含了多少行数据
            file_record.append({'filename': file_name, 'samples': matrix_data.shape[0]})
            print(f" 完成 (样本数: {matrix_data.shape[0]})")
        else:
            print(" 跳过")

    # 合并所有数据
    if all_data_list:
        # 使用 vstack 在第0维堆叠
        final_dataset = np.vstack(all_data_list)

        print(f"\n--- 处理完毕 ---")
        print(f"总计文件数: {len(all_data_list)}")
        print(f"总样本形状: {final_dataset.shape} (样本总数, 5, 9)")

        # 保存为 .npy 文件
        np.save(output_file, final_dataset)
        print(f"数据已保存至: {output_file}")

        # (可选) 保存一份文件记录到 csv，方便你知道哪几行属于哪个文件
        record_df = pd.DataFrame(file_record)
        record_df.to_csv(output_file.replace('.npy', '_log.csv'), index=False)
        print(f"文件记录已保存至: {output_file.replace('.npy', '_log.csv')}")

    else:
        print("没有有效的数据被处理。")


if __name__ == "__main__":
    # ================= 配置区域 =================

    # 1. 设置包含Excel文件的文件夹路径 ('.' 代表当前目录)
    my_input_folder = 'data/VFT/ADHD'

    # 2. 设置输出文件名 (.npy格式)
    my_output_file = 'data/VFT/ADHD_grid.npy'

    # ===========================================

    # 确保文件夹存在
    if not os.path.exists(my_input_folder):
        print(f"错误：文件夹 '{my_input_folder}' 不存在，请创建并放入Excel文件。")
    else:
        batch_process_folder(my_input_folder, my_output_file)