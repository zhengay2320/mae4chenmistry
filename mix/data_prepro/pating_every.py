import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
# 输入和输出路径
input_dir = "/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/ir_json_dataset/train"      # 存放json文件的文件夹
output_dir = "/home/ubuntu/data/zhenganyang/MAE_data/mix/raw_data/output_plots"        # 图片保存文件夹

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 遍历所有json文件
for filename in os.listdir(input_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(input_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 确保是列表
            if not isinstance(data, list) or len(data) == 0:
                print(f"跳过 {filename}：不是有效列表")
                continue

            # 随机选一条
            sample = random.choice(data)

            # 提取 ir_spectra
            spectra = sample.get("ir_spectra", None)

            if spectra is None or len(spectra) == 0:
                print(f"跳过 {filename}：没有 ir_spectra")
                continue
             # 保存图片
            output_path = os.path.join(
                    output_dir, filename.replace(".json", ".png")
                )

            # 绘图
            plt.figure(figsize=(20, 4))

            spectra_plot = spectra[::3]

            x = np.linspace(400, 4000, len(spectra_plot))

            plt.plot(x, spectra_plot, linewidth=1)

            plt.xlabel("Wavenumber (cm⁻¹)")
            plt.ylabel("Intensity")
            plt.title(f"IR Spectra from {filename}")

            plt.grid(alpha=0.3)
            plt.gca().invert_xaxis()

            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"已保存: {output_path}")

        except Exception as e:
            print(f"处理 {filename} 出错: {e}")
