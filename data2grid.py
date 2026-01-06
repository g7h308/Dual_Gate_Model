import pandas as pd
import numpy as np
import os
import glob
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata


# ==========================================
# 1. 核心处理函数 (修改：增加 sheet_name 参数)
# ==========================================
def process_single_sheet(file_path, sheet_name):
    """
    处理单个 Excel 文件的指定 Sheet，返回 (T, 5, 9) 的矩阵数组
    """
    try:
        # 修改：指定读取 sheet_name
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"    - [读取失败] Sheet: {sheet_name} - {e}")
        return None

    data = df.values
    # 检查列数，至少需要22列传感器数据
    if data.shape[1] < 22:
        print(f"    - [跳过] 列数不足: {sheet_name}")
        return None

    sensor_data = data[:, :22]

    # 标准化 (针对当前Sheet的数据独立标准化)
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(sensor_data)

    # 坐标映射
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

    # 插值处理
    for row_values in data_norm:
        grid_z = griddata(points, row_values, (grid_x, grid_y), method='cubic')
        # 处理 NaN (用最近邻填充)
        if np.isnan(grid_z).any():
            grid_z_nearest = griddata(points, row_values, (grid_x, grid_y), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

        # 角落修正
        grid_z[0, 0] = (grid_z[0, 1] + grid_z[1, 0]) / 2.0
        grid_z[0, 8] = (grid_z[0, 7] + grid_z[1, 8]) / 2.0
        grid_z[4, 0] = (grid_z[3, 0] + grid_z[4, 1]) / 2.0
        grid_z[4, 8] = (grid_z[3, 8] + grid_z[4, 7]) / 2.0

        processed_matrices.append(grid_z)

    return np.array(processed_matrices)


def process_single_sheet_zero(file_path, sheet_name):
    """
    处理单个 Excel 文件的指定 Sheet
    修改：不进行插值，空白处直接填 0
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"    - [读取失败] Sheet: {sheet_name} - {e}")
        return None

    data = df.values
    # 检查列数
    if data.shape[1] < 22:
        print(f"    - [跳过] 列数不足: {sheet_name}")
        return None

    sensor_data = data[:, :22]

    # 标准化 (依然保留，保证数据量级统一)
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(sensor_data)

    # 坐标映射 (保持不变)
    coords_map = {
        0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
        4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
        9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
        13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
        18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
    }

    processed_matrices = []

    # ------------------ 修改开始 ------------------
    for row_values in data_norm:
        # 1. 创建一个 5x9 的全 0 矩阵
        grid_z = np.zeros((5, 9))

        # 2. 将 22 个传感器的数据填入对应位置
        for i in range(22):
            if i in coords_map:
                r, c = coords_map[i]
                grid_z[r, c] = row_values[i]

        # (已删除插值、NaN处理和角落修正代码)

        processed_matrices.append(grid_z)
    # ------------------ 修改结束 ------------------

    return np.array(processed_matrices)


# ==========================================
# 2. 辅助函数：对齐与保存
# ==========================================
def align_and_save(data_list, output_path, target_len, type_name="Data"):
    """
    对数据列表进行 Padding/Truncating 并保存为 .npy
    """
    print(f"\n--- 处理 {type_name} 数据 ---")

    padded_data_list = []
    for d in data_list:
        current_len = d.shape[0]

        if current_len < target_len:
            # 填充 (Padding)
            pad_width = target_len - current_len
            d_padded = np.pad(d, ((0, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=0)
            padded_data_list.append(d_padded)

        elif current_len > target_len:
            # 截断 (Truncating)
            d_truncated = d[:target_len, :, :]
            padded_data_list.append(d_truncated)

        else:
            padded_data_list.append(d)

    final_dataset = np.array(padded_data_list)
    print(f"形状: {final_dataset.shape}")
    np.save(output_path, final_dataset)
    print(f"已保存: {output_path}")


# ==========================================
# 3. 批量处理主程序 (双 Sheet 版本)
# ==========================================
def batch_process_folder_dual(input_folder, output_base_name, fixed_length=None):
    """
    input_folder: 输入文件夹
    output_base_name: 输出文件基础路径 (不带 .npy)，程序会自动添加 _oxy.npy 和 _dxy.npy
    """
    print(f"--- 开始处理文件夹: {input_folder} ---")
    all_files = glob.glob(os.path.join(input_folder, "*.xlsx")) + \
                glob.glob(os.path.join(input_folder, "*.xls"))

    if not all_files:
        print("未找到Excel文件！")
        return

    # 分别存储两个种类的数据
    oxy_list = []
    dxy_list = []

    valid_count = 0

    # 1. 读取所有文件
    for i, file_path in enumerate(all_files):
        file_name = os.path.basename(file_path)
        print(f"[{i + 1}/{len(all_files)}] 读取: {file_name}")

        # 分别处理两个 sheet
        oxy_data = process_single_sheet_zero(file_path, 'oxyData')
        dxy_data = process_single_sheet_zero(file_path, 'dxyData')

        # 只有当两个 sheet 都成功读取时，才保留该样本
        if oxy_data is not None and dxy_data is not None:
            # 简单检查一下两个 sheet 的时间步长是否一致（通常应该一致）
            if oxy_data.shape[0] != dxy_data.shape[0]:
                print(f"    [警告] 时间步长不一致 (Oxy:{oxy_data.shape[0]}, Dxy:{dxy_data.shape[0]})，将各自截断/补齐。")

            oxy_list.append(oxy_data)
            dxy_list.append(dxy_data)
            valid_count += 1
            print(f"    -> 成功 (T={oxy_data.shape[0]})")
        else:
            print("    -> 跳过 (某个 Sheet 读取失败)")

    if valid_count == 0:
        print("没有有效数据。")
        return

    # 2. 确定目标长度
    # 策略：在所有有效数据(oxy 和 dxy)中找到最大的时间步长，或者使用 fixed_length
    if fixed_length is None:
        max_len_oxy = max([d.shape[0] for d in oxy_list])
        max_len_dxy = max([d.shape[0] for d in dxy_list])
        target_len = max(max_len_oxy, max_len_dxy)
        print(f"\n--- 自动对齐长度: {target_len} (Based on Max T) ---")
    else:
        target_len = fixed_length
        print(f"\n--- 强制对齐长度: {target_len} ---")

    # 3. 分别处理并保存两个文件
    # 构造输出文件名
    # 如果 output_base_name 是 "data/VFT/HC_grid.npy"，去掉扩展名再加后缀
    base, _ = os.path.splitext(output_base_name)

    out_oxy = f"{base}_oxy_zero.npy"
    out_dxy = f"{base}_dxy_zero.npy"

    align_and_save(oxy_list, out_oxy, target_len, "OxyData")
    align_and_save(dxy_list, out_dxy, target_len, "DxyData")

    print("\n--- 全部完成 ---")


if __name__ == "__main__":
    my_input_folder = 'data/VFT/HC_xlsx'
    # 这里只写基础文件名，程序会自动生成 HC_grid_oxy.npy 和 HC_grid_dxy.npy
    my_output_base = 'data/VFT/HC_grid.npy'

    if os.path.exists(my_input_folder):
        batch_process_folder_dual(my_input_folder, my_output_base, fixed_length=None)
    else:
        print(f"文件夹不存在: {my_input_folder}")