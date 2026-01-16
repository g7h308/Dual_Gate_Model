import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_and_plot_normalized(folder_path, plot_limit=2):
    # ==========================
    # 1. 读取数据 (与之前相同)
    # ==========================
    file_pattern = os.path.join(folder_path, "*.xlsx")
    files = sorted(glob.glob(file_pattern))

    if len(files) == 0:
        print("未找到xlsx文件，请检查路径。")
        return

    print(f"找到 {len(files)} 个文件，开始读取...")

    data_list = []

    for i, file in enumerate(files):
        try:
            # 读取 oxyData sheet
            df = pd.read_excel(file, sheet_name='oxyData', engine='openpyxl')

            # 提取前 1600 行 (去掉最后一行)，保留所有列
            df_cleaned =df

            # 存入列表
            data_list.append(df_cleaned.values)

            # 打印进度 (每5个文件打印一次，避免刷屏)
            if (i + 1) % 5 == 0:
                print(f"已处理 {i + 1}/{len(files)}...")

        except Exception as e:
            print(f"读取文件 {file} 出错: {e}")

    # ==========================
    # 2. 构建与重塑 (Reshape)
    # ==========================
    raw_data = np.array(data_list)  # (47, 1600, 22)

    # 转换为 (样本数, 通道数, 时间节点) -> (47, 22, 1600)
    final_data = np.transpose(raw_data, (0, 2, 1))

    print("-" * 30)
    print(f"原始数据形状: {final_data.shape}")

    # ==========================
    # 3. 数据标准化 (Standardization)
    # ==========================
    print("正在进行 Z-score 标准化...")

    # 计算均值和标准差
    # axis=2 表示沿着“时间”轴计算，即对每个样本的每个通道单独计算均值和方差
    # keepdims=True 保持维度为 (47, 22, 1)，以便利用广播机制直接相减相除
    means = np.mean(final_data, axis=2, keepdims=True)
    stds = np.std(final_data, axis=2, keepdims=True)

    # 防止标准差为0的情况（比如某通道全是0），将0替换为1避免报错
    stds[stds == 0] = 1

    # 执行 Z-score 标准化公式: (x - u) / sigma
    normalized_data = (final_data - means) / stds

    print(f"标准化后数据形状: {normalized_data.shape}")
    print("标准化完成 (每个时间序列均值为0，标准差为1)")
    print("-" * 30)

    # ==========================
    # 4. 绘图 (Plotting)
    # ==========================
    time_steps = np.arange(normalized_data.shape[2])

    num_channels_to_plot = plot_limit if plot_limit else normalized_data.shape[1]

    print(f"正在绘制前 {num_channels_to_plot} 个通道的图像...")

    for ch in range(num_channels_to_plot):
        plt.figure(figsize=(12, 6))

        # 提取当前通道所有样本的数据 (47, 1600)
        channel_data = normalized_data[:, ch, :]

        # 转置为 (1600, 47) 以便 plot 函数按列绘制47条线
        plt.plot(time_steps, channel_data.T, alpha=0.4, linewidth=0.8)

        # 添加一条粗的平均线（黑色），方便观察整体趋势
        mean_trend = np.mean(channel_data, axis=0)
        plt.plot(time_steps, mean_trend, color='black', linewidth=2, linestyle='--', label='Average Trend')

        plt.title(f'Channel {ch + 1} (Z-score Standardized)')
        plt.xlabel('Time Points')
        plt.ylabel('Standardized Amplitude (Z-score)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.show()


def main_pipeline(folder_path, plot_limit=2):
    """
    全流程处理函数：
    1. 读取文件夹内所有xlsx
    2. 提取 oxyData, 去掉最后一行
    3. 转换为 (47, 22, 1600) 并标准化
    4. 绘制脊线图
    """

    # =========================================================
    # 第一步：读取与清洗
    # =========================================================
    file_pattern = os.path.join(folder_path, "*.xlsx")
    files = sorted(glob.glob(file_pattern))

    if len(files) == 0:
        print(f"错误：在路径 '{folder_path}' 下未找到 .xlsx 文件。")
        return

    print(f"检测到 {len(files)} 个文件，开始读取数据...")

    data_list = []

    for i, file in enumerate(files):
        try:
            # 读取指定 sheet
            df = pd.read_excel(file, sheet_name='oxyData', engine='openpyxl')

            # 提取前 1600 行 (去掉最后一行)
            df_cleaned = df.iloc[:-1, :]

            # 存入列表，此时形状为 (1600, 22)
            data_list.append(df_cleaned.values)

            # 简单的进度提示
            if (i + 1) % 10 == 0:
                print(f"  - 已读取 {i + 1} / {len(files)} 个文件")

        except Exception as e:
            print(f"  ! 读取文件 {os.path.basename(file)} 失败: {e}")

    # =========================================================
    # 第二步：形状变换与标准化
    # =========================================================
    if not data_list:
        print("未读取到有效数据。")
        return

    # 堆叠成 (样本数, 时间步, 通道数) -> (47, 1600, 22)
    raw_data = np.array(data_list)

    # 转置为 (样本数, 通道数, 时间步) -> (47, 22, 1600)
    final_data = np.transpose(raw_data, (0, 2, 1))

    print("-" * 30)
    print(f"数据矩阵构建完成，形状: {final_data.shape}")
    print("正在进行 Z-score 标准化...")

    # 计算均值和标准差 (针对每个样本的每个通道独立计算)
    means = np.mean(final_data, axis=2, keepdims=True)
    stds = np.std(final_data, axis=2, keepdims=True)
    stds[stds == 0] = 1  # 避免除以0

    # 标准化公式
    normalized_data = (final_data - means) / stds

    # =========================================================
    # 第三步：绘制脊线图 (Ridge Plot)
    # =========================================================
    time_steps = np.arange(normalized_data.shape[2])
    num_samples = normalized_data.shape[0]

    # 确定要画几个通道
    channels_to_plot = plot_limit if plot_limit else normalized_data.shape[1]

    print(f"开始绘图... 将展示前 {channels_to_plot} 个通道的脊线图。")

    for ch in range(channels_to_plot):
        # 提取当前通道数据: (47, 1600)
        data_channel = final_data[:, ch, :]

        # 设置画布大小 (高度可以适当增加以容纳47行)
        fig, ax = plt.subplots(figsize=(12, 14))

        # 偏移量控制：决定行与行之间的间距
        # 因为数据是标准化后的（约±3范围），取 2-3 左右比较合适
        offset_step = 2.0

        # 循环绘制每一条线
        for i in range(num_samples):
            series = data_channel[i, :]

            # 计算这一行的 Y 轴基准位置
            base_y = i * offset_step
            # 实际绘制的 Y 值 = 原始数据 + 基准位置
            plot_y = series + base_y

            # 技巧：Zorder (图层顺序)
            # 让下方的图层(i小) 覆盖 上方的图层(i大) -> 看起来像从前往后排列
            # 或者反过来。这里我们让 i 越大越靠后(zorder越小)，这样前面的山峰会挡住后面的
            z_order = num_samples - i

            # 1. 填充白色背景 (用于遮挡后面的线，制造层叠感)
            ax.fill_between(time_steps, base_y, plot_y, color='white', zorder=z_order)

            # 2. 绘制波形线
            # 颜色可以随 i 渐变，也可以纯黑。这里用纯黑显得干净。
            ax.plot(time_steps, plot_y, color='black', linewidth=0.7, zorder=z_order + 0.1)

        # 设置图表美化
        ax.set_title(f'Channel {ch + 1} Ridge Plot (47 Samples)', fontsize=14, pad=20)
        ax.set_xlabel('Time Points (0-1600)', fontsize=12)
        ax.set_ylabel('Sample ID', fontsize=12)

        # 设置 Y 轴刻度，只显示样本编号
        yticks = np.arange(0, num_samples * offset_step, offset_step)
        ax.set_yticks(yticks)
        # 字体设小一点，因为有47个标签
        ax.set_yticklabels([f"S{i + 1}" for i in range(num_samples)], fontsize=7)

        # 去掉顶部和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)  # 左侧边框也可去掉，只留刻度

        # 调整布局
        plt.tight_layout()
        plt.show()



# ==========================================
# 设置路径并运行
# ==========================================
folder_path = r"./data/VFT/HC_xlsx"

# 运行处理，只画前2张
process_and_plot_normalized(folder_path, plot_limit=2)






