import torch
from models import mae_model # 请确保模型路径正确
from datasets.ir_spectra_dataset import IRSpectraDataset  # 请确保数据集路径正确
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from torch.utils.data import DataLoader
import numpy as np

import logging

# 设置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),
                        logging.FileHandler('test_log.txt')  # 将日志同时输出到控制台和文件
                    ])
logger = logging.getLogger()


# 加载检查点（如果有的话）
def load_model_from_checkpoint(model, checkpoint_path):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, using randomly initialized model.")
    return model


def calculate_ssim(original, reconstructed):
    """计算结构相似性指数（SSIM）"""
    # Ensure the data is 2D by adding a singleton dimension if necessary
    original = original.unsqueeze(0).cpu().numpy()  # Add batch dimension if needed
    reconstructed = reconstructed.unsqueeze(0).cpu().numpy()  # Add batch dimension if needed

    # If the data is 1D, expand to 2D (if it's not already)
    original = np.expand_dims(original, axis=0)  # Add a second dimension for SSIM
    reconstructed = np.expand_dims(reconstructed, axis=0)  # Add a second dimension for SSIM

    # Set win_size to a smaller value like 3 for 1D data or adjust it as necessary
    return ssim(original, reconstructed, data_range=reconstructed.max() - reconstructed.min(), win_size=3,
                channel_axis=0)


def visualize_results_with_mask(original, mask, reconstructed, idx=0, epoch=0):
    """绘制原始数据曲线并填充掩码区域（优化清晰度版本）"""

    # ===== 可调参数 =====
    step = 3  # 降采样步长（建议2~5）
    figsize = (20, 4)

    # 转 numpy
    original_np = original[idx].cpu().numpy()
    reconstructed_np = reconstructed[idx].cpu().numpy()
    masked_region = mask[idx].cpu().numpy() == 1

    # ===== 构造 full mask =====
    full_masked_region = np.zeros(len(original_np), dtype=bool)
    patch_size = len(original_np) // len(masked_region)

    for i in range(len(masked_region)):
        if masked_region[i]:
            start = i * patch_size
            end = start + patch_size
            full_masked_region[start:end] = True

    # ===== 降采样 =====
    original_plot = original_np[::step]
    reconstructed_plot = reconstructed_np[::step]
    mask_plot = full_masked_region[::step]

    # ===== 构造波数轴（更专业）=====
    x = np.linspace(400, 4000, len(original_plot))

    # ===== 绘图 =====
    plt.figure(figsize=figsize)

    plt.plot(x, original_plot, label='Original Spectrum', linewidth=1)
    plt.plot(x, reconstructed_plot, label='Reconstructed Spectrum', linestyle='dotted', linewidth=1)

    # mask 区域填充
    plt.fill_between(
        x,
        original_plot,
        where=mask_plot,
        color='orange',
        alpha=0.3,
        label='Masked Regions'
    )

    # ===== 图像优化 =====
    plt.title(f"Spectrum {idx} (Masked Regions)")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")

    plt.grid(alpha=0.3)
    plt.legend()

    # IR 常见：x轴反转
    plt.gca().invert_xaxis()

    # ===== 保存 =====
    output_dir = f"./output/test_results{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        f"{output_dir}/spectrum_{idx}_with_mask.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()


# 测试流程
def test_model(model, test_loader, epoch=60):
    model.eval()  # 进入评估模式
    total_mse = 0
    num_samples = 0
    logger.info(f"Starting epoch {epoch} testing...")
    j = 0
    for idx, batch in enumerate(test_loader):
        batch = batch.cuda()

        with torch.no_grad():
            loss, pred, mask = model(batch, mask_ratio=0.3)
            outputs = model.unpatchify(pred)

        # 计算 SSIM 和 MSE
        # batch_ssim = 0
        batch_mse = 0
        for i in range(batch.size(0)):
            original = batch[i]
            reconstructed = outputs[i]
            # batch_ssim += calculate_ssim(original, reconstructed)
            batch_mse += torch.mean((original - reconstructed) ** 2)

        # batch_ssim /= batch.size(0)
        batch_mse /= batch.size(0)
        # total_ssim += batch_ssim
        total_mse += batch_mse

        # 可视化
        if j < 100:  # 选择前20个样本进行可视化
            sample_idx = 0
            visualize_results_with_mask(batch, mask, outputs, idx=idx, epoch=epoch)
        j += 1

        num_samples += batch.size(0)

    # avg_ssim = total_ssim / num_samples
    # avg_mse = total_mse / num_samples
    # print(f"第{epoch}Test SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.4f}")
    logger.info(f"第{epoch}Test:MSE: {total_mse:.4f}")


# 执行测试

if __name__ == '__main__':
    # 配置
    test_dataset_path = r'/home/rspip/zay/ir_json_dataset/train.pkl'  # 替换为测试集路径
    checkpoint_path = r'/home/rspip/zay/mae/mae-main/output_dir/checkpoint-60.pth'  # 替换为模型检查点路径，如果为空则随机初始化模型
    output_dir = './output/test_results'  # 保存输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 模型定义
    model = mae_model.mae_vit_base_patch16_dec512d8b()  # 使用你的模型结构

    model = load_model_from_checkpoint(model, checkpoint_path)
    model = model.cuda()  # 将模型移动到GPU

    # 数据集加载
    test_dataset = IRSpectraDataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    test_model(model, test_loader)
