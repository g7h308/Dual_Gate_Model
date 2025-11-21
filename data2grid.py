import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import griddata


def process_sensor_data(file_path):
    """
    读取Excel，标准化数据，并将其转换为 5x9 插值矩阵
    """
    print(f"正在读取文件: {file_path} ...")

    # 1. 读取数据
    # 如果第一行是标题，请去掉 header=None；如果没有标题，保留 header=None
    try:
        df = pd.read_excel(file_path, header=None)
    except FileNotFoundError:
        print("错误：找不到文件，请检查路径。")
        return None

    data = df.values

    # 检查列数
    if data.shape[1] < 22:
        print(f"错误：数据列数不足22列 (当前: {data.shape[1]})")
        return None

    # 截取前22列
    sensor_data = data[:, :22]

    # 2. 标准化 (Z-score)
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(sensor_data)

    # 3. 定义坐标映射 (0-21)
    coords_map = {
        0: (0, 1), 1: (0, 3), 2: (0, 5), 3: (0, 7),
        4: (1, 0), 5: (1, 2), 6: (1, 4), 7: (1, 6), 8: (1, 8),
        9: (2, 1), 10: (2, 3), 11: (2, 5), 12: (2, 7),
        13: (3, 0), 14: (3, 2), 15: (3, 4), 16: (3, 6), 17: (3, 8),
        18: (4, 1), 19: (4, 3), 20: (4, 5), 21: (4, 7)
    }

    # 提取坐标点
    points = np.array([coords_map[i] for i in range(22)])
    # 创建 5x9 网格
    grid_x, grid_y = np.mgrid[0:5:1, 0:9:1]

    processed_matrices = []

    print(f"正在处理 {len(data_norm)} 行数据...")

    # 4. 逐行插值
    for row_values in data_norm:
        # 第一步：三次样条插值 (平滑，但边缘可能是NaN)
        grid_z = griddata(points, row_values, (grid_x, grid_y), method='cubic')

        # 第二步：填充NaN (使用最近邻插值填补角落)
        if np.isnan(grid_z).any():
            grid_z_nearest = griddata(points, row_values, (grid_x, grid_y), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_nearest[np.isnan(grid_z)]

        processed_matrices.append(grid_z)

    final_output = np.array(processed_matrices)
    print("处理完成。")
    return final_output


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请在这里修改为你的真实文件名
    excel_file = 'data.xlsx'
    # ===========================================

    result = process_sensor_data(excel_file)

    if result is not None:
        print(f"输出矩阵形状: {result.shape} (样本数, 5, 9)")

        # 可选：保存为 numpy 文件，方便后续读取
        # np.save('processed_data.npy', result)

        # 可视化检查第一行数据（确保转换逻辑正确）
        plt.imshow(result[0], cmap='jet', origin='upper')
        plt.colorbar()
        plt.title('Row 0: 5x9 Matrix Visualization')
        plt.show()