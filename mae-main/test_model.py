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

def visualize_results_with_mask(original, mask, reconstructed, idx=0, save_id=0, epoch=0):
    step = 3
    figsize = (20, 4)

    original_np = original[idx].detach().cpu().numpy()
    reconstructed_np = reconstructed[idx].detach().cpu().numpy()
    masked_region = mask[idx].detach().cpu().numpy() == 1

    full_masked_region = np.zeros(len(original_np), dtype=bool)
    patch_size = len(original_np) // len(masked_region)

    for i in range(len(masked_region)):
        if masked_region[i]:
            start = i * patch_size
            end = start + patch_size
            full_masked_region[start:end] = True

    original_plot = original_np[::step]
    reconstructed_plot = reconstructed_np[::step]
    mask_plot = full_masked_region[::step]

    x = np.linspace(400, 4000, len(original_plot))

    plt.figure(figsize=figsize)
    plt.plot(x, original_plot, label='Original Spectrum', linewidth=1)
    plt.plot(x, reconstructed_plot, label='Reconstructed Spectrum', linestyle='dotted', linewidth=1)

    plt.fill_between(
        x,
        original_plot,
        where=mask_plot,
        color='orange',
        alpha=0.3,
        label='Masked Regions'
    )

    plt.title(f"Spectrum {save_id} (Masked Regions)")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.gca().invert_xaxis()

    output_dir = f"./output/test_results{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        f"{output_dir}/spectrum_{save_id:04d}_with_mask.png",
        dpi=150,
        bbox_inches='tight'
    )
    plt.close()


# 测试流程
def test_model(model, test_loader, epoch=304):
    model.eval()
    total_masked_mse = 0.0
    num_batches = 0
    logger.info(f"Starting epoch {epoch} testing...")


    for batch_idx, batch in enumerate(test_loader):
        batch = batch.cuda()

        with torch.no_grad():
            loss, pred, mask = model(batch, mask_ratio=0.3)

            target = model.patchify(batch)                     # [N, L, p]
            mask_expanded = mask.unsqueeze(-1)                # [N, L, 1]

            pasted = target * (1 - mask_expanded) + pred * mask_expanded
            outputs = model.unpatchify(pasted)

            mse_per_patch = ((pred - target) ** 2).mean(dim=-1)   # [N, L]
            masked_mse = (mse_per_patch * mask).sum() / mask.sum()

        total_masked_mse += masked_mse.item()
        num_batches += 1

        if batch_idx < 100:
            print(f'绘制第{batch_idx}张图')

            visualize_results_with_mask(
                batch, mask, outputs,
                idx=0,
                save_id=batch_idx,
                epoch=epoch
            )


    avg_masked_mse = total_masked_mse / max(num_batches, 1)
    logger.info(f"Epoch {epoch} Test Masked MSE: {avg_masked_mse:.6f}")

# 执行测试

if __name__ == '__main__':
    # 配置
    test_dataset_path = r'/home/rspip/zay/ir_json_dataset/train.pkl'  # 替换为测试集路径
    checkpoint_path = r'/home/rspip/zay/mae/mae-main/output_dir/checkpoint-304.pth'  # 替换为模型检查点路径，如果为空则随机初始化模型
    output_dir = './output/test_results'  # 保存输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 模型定义
    model = mae_model.mae_vit_base_patch9_dec512d8b()  # 使用你的模型结构

    model = load_model_from_checkpoint(model, checkpoint_path)
    model = model.cuda()  # 将模型移动到GPU

    # 数据集加载
    test_dataset = IRSpectraDataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_model(model, test_loader)
