import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from test import find_peaks_joint
from denoise import STMDenoiser
from scipy.spatial import Delaunay
from tqdm import tqdm
import json
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.optimize import curve_fit

# 使用与infer.py相同的配置
grid_size = 14
ratio = 0.714
HISTOGRAM_BINS = 200  # 全局变量控制直方图的bin数量
CONFIG = {
    'TILE_SIZE': 1024,
    'OVERLAP_THRESHOLD': int(grid_size * ratio),
    'MIN_DISTANCE': int(grid_size * ratio),
    'GRID_SIZE': grid_size,
    'MIN_THRESH': 0.01,
    'NMS_KSIZE': int(grid_size * ratio),
    'PEAK_MIN_DISTANCE': int(grid_size * ratio),
    'SCALE_FACTOR': 2,
    'RESIZE_TO': None,
    'OUTPUT_DIR': '../raw/classical_results',
    'INPUT_DIR': '../raw/target/images/BIG/1-1'
}

def gaussian(x, a, mu, sigma):
    """单高斯函数"""
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2))

def double_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
    """双高斯函数"""
    return gaussian(x, a1, mu1, sigma1) + gaussian(x, a2, mu2, sigma2)

def fit_gaussians(heights, bins=HISTOGRAM_BINS):
    """
    对高度分布进行高斯拟合
    Returns:
        fit_params: 拟合参数 [a1, mu1, sigma1, a2, mu2, sigma2]
        fit_curve: 拟合曲线
        bin_centers: 直方图中心点
        atom_counts: 拟合得到的原子数量 [count1, count2]
    """
    # 计算直方图
    hist, bin_edges = np.histogram(heights, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 计算数据的统计特性
    mean = np.mean(heights)
    std = np.std(heights)
    max_val = np.max(hist)
    
    # 改进初始参数估计
    # 假设第一个峰在均值以下，第二个峰在均值以上
    p0 = [
        max_val/2,  # a1
        mean - std/2,  # mu1
        std/2,  # sigma1
        max_val/2,  # a2
        mean + std/2,  # mu2
        std/2   # sigma2
    ]
    
    # 设置参数边界
    bounds = (
        [0, mean-2*std, 0, 0, mean, 0],  # 下界
        [max_val*2, mean, std*2, max_val*2, mean+2*std, std*2]  # 上界
    )
    
    # 拟合双高斯
    try:
        popt, pcov = curve_fit(double_gaussian, bin_centers, hist, p0=p0, bounds=bounds)
        fit_curve = double_gaussian(bin_centers, *popt)
        
        # 计算每个高斯峰下的面积（即原子数量）
        # 由于直方图是归一化的，需要乘以总原子数
        total_atoms = len(heights)
        bin_width = bin_edges[1] - bin_edges[0]
        area1 = np.sum(gaussian(bin_centers, popt[0], popt[1], popt[2])) * bin_width * total_atoms
        area2 = np.sum(gaussian(bin_centers, popt[3], popt[4], popt[5])) * bin_width * total_atoms
        
        return popt, fit_curve, bin_centers, [area1, area2]
    except RuntimeError as e:
        print(f"高斯拟合失败: {str(e)}")
        return None, None, bin_centers, None

def high_pass_filter(image, cutoff_wavelength_nm):
    """
    对图像进行高通滤波
    Args:
        image: 输入图像
        cutoff_wavelength_nm: 截止波长（nm）
    Returns:
        滤波后的图像
    """
    # 获取图像尺寸
    height, width = image.shape
    
    # 计算频率网格
    u = np.fft.fftfreq(width)
    v = np.fft.fftfreq(height)
    u, v = np.meshgrid(u, v)
    
    # 计算空间频率（nm^-1）
    freq = np.sqrt(u**2 + v**2) / 100
    
    # 创建高通滤波器（使用高斯过渡）
    sigma = 0.0001  # 控制过渡的平滑程度
    filter_mask = 1 - np.exp(-(freq / (1/cutoff_wavelength_nm))**2 / (2 * sigma**2))
    
    # 进行傅里叶变换
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)
    
    # 应用滤波器
    filtered_fft = fft_image_shifted * filter_mask
    
    # 反变换回空间域
    filtered_image = np.real(ifft2(ifftshift(filtered_fft)))
    
    # 归一化到原始图像的动态范围
    filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())
    filtered_image = filtered_image * (image.max() - image.min()) + image.min()
    
    return filtered_image

def process_image(image_path, model, device):
    """
    处理单张图像，返回检测到的原子坐标和对应的亮度值
    """
    filename = os.path.basename(image_path).split('.')[0]
    print(f"处理图像: {filename}")
    
    # 读取图像
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = original_img.copy()

    # 图像预处理
    if CONFIG['RESIZE_TO'] is not None:
        resize_width, resize_height = CONFIG['RESIZE_TO']
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        original_img = cv2.resize(original_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    # 放大图像
    if CONFIG['SCALE_FACTOR'] != 1.0:
        height, width = img.shape
        new_height = int(height * CONFIG['SCALE_FACTOR'])
        new_width = int(width * CONFIG['SCALE_FACTOR'])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        original_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 对图像进行高通滤波
    filtered_img = high_pass_filter(img, cutoff_wavelength_nm=0.67)

    # 分割图像
    tiles = []
    for i in range(0, img.shape[0], CONFIG['TILE_SIZE']):
        for j in range(0, img.shape[1], CONFIG['TILE_SIZE']):
            if i + CONFIG['TILE_SIZE'] <= img.shape[0] and j + CONFIG['TILE_SIZE'] <= img.shape[1]:
                tiles.append((img[i:i + CONFIG['TILE_SIZE'], j:j + CONFIG['TILE_SIZE']], (i, j)))

    all_coords = []
    filtered_coords = []

    # 只在原始图像上进行一次AI识别
    for tile, (x, y) in tqdm(tiles, desc="Processing original tiles"):
        tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_heatmap = model(tile_tensor).squeeze(0).cpu().numpy()

        # 归一化热力图
        pred_heatmap[0] = (pred_heatmap[0] - pred_heatmap[0].min()) / (pred_heatmap[0].max() - pred_heatmap[0].min())
        pred_heatmap[1] = (pred_heatmap[1] - pred_heatmap[1].min()) / (pred_heatmap[1].max() - pred_heatmap[1].min())

        # 寻峰
        coords = find_peaks_joint(
            te_heatmap=pred_heatmap[0],
            se_heatmap=pred_heatmap[1],
            grid_size=CONFIG['GRID_SIZE'],
            min_thresh=CONFIG['MIN_THRESH'],
            nms_ksize=CONFIG['NMS_KSIZE'],
            min_distance=CONFIG['PEAK_MIN_DISTANCE']
        )

        # 获取每个检测点的亮度值（原始图像和滤波后图像）
        for coord in coords:
            # 保存原始坐标
            original_coord = coord.copy()
            original_coord["x"] += y
            original_coord["y"] += x
            # 获取原始图像的亮度值
            height = original_img[int(original_coord["y"]), int(original_coord["x"])]
            original_coord["height"] = height
            all_coords.append(original_coord)

            # 保存滤波后图像的坐标
            filtered_coord = coord.copy()
            filtered_coord["x"] += y
            filtered_coord["y"] += x
            # 获取滤波后图像的亮度值
            height = filtered_img[int(filtered_coord["y"]), int(filtered_coord["x"])]
            filtered_coord["height"] = height
            filtered_coords.append(filtered_coord)

    return all_coords, filtered_coords

def plot_height_distributions(all_coords, filtered_coords, output_dir):
    """
    绘制原始图像和滤波后图像的高度分布直方图，并进行高斯拟合分析
    """
    # 创建图形
    plt.figure(figsize=(15, 6))
    
    # 创建数据文件
    data_file = os.path.join(output_dir, "height_distribution_data.txt")
    with open(data_file, 'w') as f:
        f.write("Height Distribution Analysis Results\n")
        f.write("==================================\n\n")
    
    # 绘制原始图像的分布
    plt.subplot(1, 2, 1)
    te_heights = [coord["height"] for coord in all_coords if coord["class"] == "Te"]
    se_heights = [coord["height"] for coord in all_coords if coord["class"] == "Se"]
    all_heights = [coord["height"] for coord in all_coords]
    
    # 计算比例
    te_count = len(te_heights)
    se_count = len(se_heights)
    total_count = te_count + se_count
    te_ratio = (te_count / total_count) * 100 if total_count > 0 else 0
    se_ratio = (se_count / total_count) * 100 if total_count > 0 else 0
    
    # 计算所有直方图共用的bin边界
    hist_all, bin_edges = np.histogram(all_heights, bins=HISTOGRAM_BINS, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 使用相同的bin边界计算Te和Se的直方图
    hist_te, _ = np.histogram(te_heights, bins=bin_edges, density=True)
    hist_se, _ = np.histogram(se_heights, bins=bin_edges, density=True)
    
    # 保存直方图数据到文件
    with open(data_file, 'a') as f:
        f.write("\nOriginal Image Data:\n")
        f.write("-------------------\n")
        f.write(f"Total atoms: {total_count}\n")
        f.write(f"Te atoms: {te_count} ({te_ratio:.1f}%)\n")
        f.write(f"Se atoms: {se_count} ({se_ratio:.1f}%)\n\n")
        f.write("Histogram Data:\n")
        f.write("Bin Center, All Atoms, Te Atoms, Se Atoms\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.6f}, {hist_all[i]:.6f}, {hist_te[i]:.6f}, {hist_se[i]:.6f}\n")
    
    # 绘制直方图
    plt.bar(bin_centers, hist_all, width=bin_edges[1]-bin_edges[0], alpha=0.3, color='gray', label='All Atoms')
    plt.bar(bin_centers, hist_te, width=bin_edges[1]-bin_edges[0], alpha=0.3, color='red', label=f'Te Atoms ({te_ratio:.1f}%)')
    plt.bar(bin_centers, hist_se, width=bin_edges[1]-bin_edges[0], alpha=0.3, color='blue', label=f'Se Atoms ({se_ratio:.1f}%)')
    
    # 对总分布进行高斯拟合
    fit_params, fit_curve, bin_centers, fit_counts = fit_gaussians(all_heights, bins=HISTOGRAM_BINS)
    if fit_params is not None:
        # 绘制拟合曲线
        plt.plot(bin_centers, fit_curve, 'k-', linewidth=2, label='Gaussian Fit')
        
        # 绘制单个高斯分量
        g1 = gaussian(bin_centers, fit_params[0], fit_params[1], fit_params[2])
        g2 = gaussian(bin_centers, fit_params[3], fit_params[4], fit_params[5])
        plt.plot(bin_centers, g1, 'r--', linewidth=1, label='Gaussian 1')
        plt.plot(bin_centers, g2, 'b--', linewidth=1, label='Gaussian 2')
        
        # 计算拟合得到的原子比例
        fit_total = fit_counts[0] + fit_counts[1]
        fit_te_ratio = (fit_counts[0] / fit_total) * 100
        fit_se_ratio = (fit_counts[1] / fit_total) * 100
        
        # 在图上添加文本说明
        plt.text(0.02, 0.95, f'AI Detection:\nTe: {te_count} ({te_ratio:.1f}%)\nSe: {se_count} ({se_ratio:.1f}%)',
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        plt.text(0.02, 0.80, f'Gaussian Fit:\nTe: {fit_counts[0]:.0f} ({fit_te_ratio:.1f}%)\nSe: {fit_counts[1]:.0f} ({fit_se_ratio:.1f}%)',
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        
        # 保存拟合数据到文件
        with open(data_file, 'a') as f:
            f.write("\nGaussian Fit Results:\n")
            f.write("--------------------\n")
            f.write(f"Gaussian 1: amplitude={fit_params[0]:.3f}, mean={fit_params[1]:.3f}, std={fit_params[2]:.3f}\n")
            f.write(f"Gaussian 2: amplitude={fit_params[3]:.3f}, mean={fit_params[4]:.3f}, std={fit_params[5]:.3f}\n")
            f.write(f"Fitted Te atoms: {fit_counts[0]:.0f} ({fit_te_ratio:.1f}%)\n")
            f.write(f"Fitted Se atoms: {fit_counts[1]:.0f} ({fit_se_ratio:.1f}%)\n")
            f.write(f"R² = {r_squared:.3f}\n")
        
        # 打印拟合参数
        print("\n原始图像高斯拟合参数:")
        print(f"高斯峰1: 振幅={fit_params[0]:.3f}, 均值={fit_params[1]:.3f}, 标准差={fit_params[2]:.3f}")
        print(f"高斯峰2: 振幅={fit_params[3]:.3f}, 均值={fit_params[4]:.3f}, 标准差={fit_params[5]:.3f}")
        print(f"拟合得到的原子数量: Te={fit_counts[0]:.0f}, Se={fit_counts[1]:.0f}")
        print(f"AI检测的原子数量: Te={te_count}, Se={se_count}")
        
        # 计算拟合优度
        ss_res = np.sum((fit_curve - hist_all)**2)
        ss_tot = np.sum((hist_all - np.mean(hist_all))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"拟合优度 R² = {r_squared:.3f}")
    
    plt.title('Original Image Height Distribution')
    plt.xlabel('Height/Brightness')
    plt.ylabel('Normalized Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 绘制滤波后图像的分布
    plt.subplot(1, 2, 2)
    te_heights = [coord["height"] for coord in filtered_coords if coord["class"] == "Te"]
    se_heights = [coord["height"] for coord in filtered_coords if coord["class"] == "Se"]
    all_heights = [coord["height"] for coord in filtered_coords]
    
    # 计算比例
    te_count = len(te_heights)
    se_count = len(se_heights)
    total_count = te_count + se_count
    te_ratio = (te_count / total_count) * 100 if total_count > 0 else 0
    se_ratio = (se_count / total_count) * 100 if total_count > 0 else 0
    
    # 计算所有直方图共用的bin边界
    hist_all, bin_edges = np.histogram(all_heights, bins=HISTOGRAM_BINS, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 使用相同的bin边界计算Te和Se的直方图
    hist_te, _ = np.histogram(te_heights, bins=bin_edges, density=True)
    hist_se, _ = np.histogram(se_heights, bins=bin_edges, density=True)
    
    # 保存直方图数据到文件
    with open(data_file, 'a') as f:
        f.write("\nFiltered Image Data:\n")
        f.write("-------------------\n")
        f.write(f"Total atoms: {total_count}\n")
        f.write(f"Te atoms: {te_count} ({te_ratio:.1f}%)\n")
        f.write(f"Se atoms: {se_count} ({se_ratio:.1f}%)\n\n")
        f.write("Histogram Data:\n")
        f.write("Bin Center, All Atoms, Te Atoms, Se Atoms\n")
        for i in range(len(bin_centers)):
            f.write(f"{bin_centers[i]:.6f}, {hist_all[i]:.6f}, {hist_te[i]:.6f}, {hist_se[i]:.6f}\n")
    
    # 绘制直方图
    plt.bar(bin_centers, hist_all, width=bin_edges[1]-bin_edges[0], alpha=0.3, color='gray', label='All Atoms')
    plt.bar(bin_centers, hist_te, width=bin_edges[1]-bin_edges[0], alpha=0.3, color='red', label=f'Te Atoms ({te_ratio:.1f}%)')
    plt.bar(bin_centers, hist_se, width=bin_edges[1]-bin_edges[0], alpha=0.3, color='blue', label=f'Se Atoms ({se_ratio:.1f}%)')
    
    # 对滤波后图像的总分布进行高斯拟合
    fit_params, fit_curve, bin_centers, fit_counts = fit_gaussians(all_heights, bins=HISTOGRAM_BINS)
    if fit_params is not None:
        # 绘制拟合曲线
        plt.plot(bin_centers, fit_curve, 'k-', linewidth=2, label='Gaussian Fit')
        
        # 绘制单个高斯分量
        g1 = gaussian(bin_centers, fit_params[0], fit_params[1], fit_params[2])
        g2 = gaussian(bin_centers, fit_params[3], fit_params[4], fit_params[5])
        plt.plot(bin_centers, g1, 'r--', linewidth=1, label='Gaussian 1')
        plt.plot(bin_centers, g2, 'b--', linewidth=1, label='Gaussian 2')
        
        # 计算拟合得到的原子比例
        fit_total = fit_counts[0] + fit_counts[1]
        fit_te_ratio = (fit_counts[0] / fit_total) * 100
        fit_se_ratio = (fit_counts[1] / fit_total) * 100
        
        # 在图上添加文本说明
        plt.text(0.02, 0.95, f'AI Detection:\nTe: {te_count} ({te_ratio:.1f}%)\nSe: {se_count} ({se_ratio:.1f}%)',
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        plt.text(0.02, 0.80, f'Gaussian Fit:\nTe: {fit_counts[0]:.0f} ({fit_te_ratio:.1f}%)\nSe: {fit_counts[1]:.0f} ({fit_se_ratio:.1f}%)',
                transform=plt.gca().transAxes, fontsize=8, verticalalignment='top')
        
        # 保存拟合数据到文件
        with open(data_file, 'a') as f:
            f.write("\nGaussian Fit Results:\n")
            f.write("--------------------\n")
            f.write(f"Gaussian 1: amplitude={fit_params[0]:.3f}, mean={fit_params[1]:.3f}, std={fit_params[2]:.3f}\n")
            f.write(f"Gaussian 2: amplitude={fit_params[3]:.3f}, mean={fit_params[4]:.3f}, std={fit_params[5]:.3f}\n")
            f.write(f"Fitted Te atoms: {fit_counts[0]:.0f} ({fit_te_ratio:.1f}%)\n")
            f.write(f"Fitted Se atoms: {fit_counts[1]:.0f} ({fit_se_ratio:.1f}%)\n")
            f.write(f"R² = {r_squared:.3f}\n")
        
        # 打印拟合参数
        print("\n滤波后图像高斯拟合参数:")
        print(f"高斯峰1: 振幅={fit_params[0]:.3f}, 均值={fit_params[1]:.3f}, 标准差={fit_params[2]:.3f}")
        print(f"高斯峰2: 振幅={fit_params[3]:.3f}, 均值={fit_params[4]:.3f}, 标准差={fit_params[5]:.3f}")
        print(f"拟合得到的原子数量: Te={fit_counts[0]:.0f}, Se={fit_counts[1]:.0f}")
        print(f"AI检测的原子数量: Te={te_count}, Se={se_count}")
        
        # 计算拟合优度
        ss_res = np.sum((fit_curve - hist_all)**2)
        ss_tot = np.sum((hist_all - np.mean(hist_all))**2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"拟合优度 R² = {r_squared:.3f}")
    
    plt.title('Filtered Image Height Distribution (0.67nm$^{-1}$)')
    plt.xlabel('Height/Brightness')
    plt.ylabel('Normalized Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, "height_distribution_plot.png")
    plt.savefig(plot_path)
    print(f"高度分布直方图已保存到: {plot_path}")
    print(f"数据已保存到: {data_file}")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load("./checkpoints_tuned/unet_tuned_final.pth", map_location=device))
    model.eval()

    input_dir = CONFIG['INPUT_DIR']
    output_dir = CONFIG['OUTPUT_DIR']
    
    # 获取文件列表并按序号排序
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(x.split('-')[0])
    )
    
    all_coords = []
    filtered_coords = []
    
    # 处理所有图像
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        coords, f_coords = process_image(image_path, model, device)
        all_coords.extend(coords)
        filtered_coords.extend(f_coords)
    
    # 绘制高度分布图
    plot_height_distributions(all_coords, filtered_coords, output_dir)

if __name__ == "__main__":
    main() 