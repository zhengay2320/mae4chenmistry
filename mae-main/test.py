import torch
from models.mae_model import MaskedAutoencoderViT1D  # 请确保模型路径正确
from datasets.ir_spectra_dataset import IRSpectraDataset  # 请确保数据集路径正确
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
from torch.utils.data import DataLoader
import numpy as np

# 配置
test_dataset_path = r'G:\test_data'  # 替换为测试集路径
checkpoint_path = None  # 替换为模型检查点路径，如果为空则随机初始化模型
output_dir = './output/test_results'  # 保存输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 模型定义
model = MaskedAutoencoderViT1D()  # 使用你的模型结构


# 加载检查点（如果有的话）
def load_model_from_checkpoint(model, checkpoint_path):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from checkpoint: {checkpoint_path}")
    else:
        print("No checkpoint found, using randomly initialized model.")
    return model


model = load_model_from_checkpoint(model, checkpoint_path)
model = model.cuda()  # 将模型移动到GPU

# 数据集加载
test_dataset = IRSpectraDataset(test_dataset_path)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


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


# def visualize_results(original, masked, reconstructed, idx=0):
#     """绘制原始数据、掩盖数据与重建数据的对比图"""
#     plt.figure(figsize=(10, 6))
#     plt.plot(original[idx].cpu().numpy(), label='Original Spectrum')
#     plt.plot(masked[idx].cpu().numpy(), label='Masked Spectrum', linestyle='dashed')
#     plt.plot(reconstructed[idx].cpu().numpy(), label='Reconstructed Spectrum', linestyle='dotted')
#     plt.title(f"Spectrum {idx}")
#     plt.legend()
#     plt.xlabel("Wavenumber")
#     plt.ylabel("Intensity")
#     plt.savefig(os.path.join(output_dir, f"spectrum_{idx}.png"))
#     plt.close()
# import matplotlib.pyplot as plt
def visualize_results_with_mask(original, mask, reconstructed, idx=0, epoch=0):
    """绘制原始数据曲线并填充掩码区域"""
    # 计算掩码区域（掩码为1的部分是被遮挡的区域）
    masked_region = mask[idx].cpu().numpy() == 1  # 掩码区域的索引

    # 初始化原始数据的掩码区域（长度为原始数据长度）
    full_masked_region = np.zeros(len(original[idx]), dtype=bool)

    # 将掩码区域扩展到与原始数据长度相同
    patch_size = original.shape[1] // mask.shape[1]  # 计算每个patch的大小
    for i in range(len(masked_region)):
        if masked_region[i]:
            start = i * patch_size
            end = start + patch_size
            full_masked_region[start:end] = True

    # 绘制原始数据
    plt.figure(figsize=(10, 6))
    plt.plot(original[idx].cpu().numpy(), label='Original Spectrum', color='gray')

    # 绘制重建数据
    plt.plot(reconstructed[idx].cpu().numpy(), label='Reconstruct Spectrum', linestyle='dotted')

    # 填充掩码区域，使用红色填充
    plt.fill_between(range(len(original[idx])), original[idx].cpu().numpy(), where=full_masked_region, color='red',
                     alpha=0.3, label='Masked Regions')

    # 添加标题和标签
    plt.title(f"Spectrum {idx} (Masked Regions)")
    plt.legend()
    plt.xlabel("Wavenumber")
    plt.ylabel("Intensity")

    # 检查目标目录是否存在，如果不存在则创建
    output_dir = f"./output/test_results{epoch}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 保存图像
    plt.savefig(f"./output/test_results{epoch}/spectrum_{idx}_with_mask.png")
    plt.close()


# 测试流程
def test_model(model, test_loader, epoch=0):
    model.eval()  # 进入评估模式
    total_ssim = 0
    total_mse = 0
    num_samples = 0

    for idx, batch in enumerate(test_loader):
        batch = batch.cuda()

        with torch.no_grad():
            oss, pred, mask = model(batch, mask_ratio=0.75)
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
        if idx < 10:  # 选择前10个样本进行可视化
            visualize_results_with_mask(batch, mask, outputs, idx, epoch)

        num_samples += batch.size(0)

    avg_ssim = total_ssim / num_samples
    avg_mse = total_mse / num_samples
    print(f"Test SSIM: {avg_ssim:.4f}, MSE: {avg_mse:.4f}")


# 执行测试
test_model(model, test_loader)
