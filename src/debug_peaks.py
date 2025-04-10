import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from denoise import STMDenoiser

# 全局配置参数
CONFIG = {
    'TILE_SIZE': 128,          # 图像分割的尺寸
    'MIN_THRESH': 0.1,         # 最小阈值
    'NMS_KSIZE': 5,            # 非极大值抑制的核大小
    'MIN_DISTANCE': 5,         # 最小距离约束
    'GRID_SIZE': 8,            # 网格大小
    'PEAK_MIN_DISTANCE': 5     # 峰值检测的最小距离
}

def simple_find_peaks(heatmap, min_thresh=0.1, nms_ksize=5, min_distance=5):
    """
    简化版的寻峰算法，只包含基本的阈值过滤和非极大值抑制。
    
    Args:
        heatmap (numpy.ndarray): 输入热力图
        min_thresh (float): 最小阈值，用于过滤低强度伪峰值
        nms_ksize (int): 非极大值抑制的核大小
        min_distance (int): 最小峰值间距
        
    Returns:
        list: 检测到的峰值坐标 [(x1, y1), (x2, y2), ...]
    """
    # 检查热力图是否全为 0
    if heatmap.max() == 0:
        return []  # 返回空列表，表示没有峰值
    
    # 创建二值掩码
    mask = (heatmap > min_thresh).astype(np.uint8)
    
    # 非极大值抑制
    max_map = cv2.dilate(heatmap, np.ones((nms_ksize, nms_ksize), np.uint8))
    peak_mask = (heatmap == max_map) & (mask == 1)
    
    # 获取峰值坐标
    peak_y, peak_x = np.where(peak_mask)
    coords = list(zip(peak_x, peak_y))
    
    # 按峰值强度排序
    coords = sorted(coords, key=lambda c: heatmap[c[1], c[0]], reverse=True)
    
    # 应用最小距离约束
    filtered_coords = []
    for x, y in coords:
        if all(np.sqrt((x - fx)**2 + (y - fy)**2) >= min_distance for fx, fy in filtered_coords):
            filtered_coords.append((x, y))
    
    return filtered_coords

def simple_find_peaks_joint(te_heatmap, se_heatmap, min_thresh=0.1, nms_ksize=5, min_distance=5):
    """
    简化版的联合寻峰算法，只包含基本的阈值过滤和非极大值抑制。
    
    Args:
        te_heatmap (numpy.ndarray): Te 原子的热力图
        se_heatmap (numpy.ndarray): Se 原子的热力图
        min_thresh (float): 最小阈值，用于过滤低强度伪峰值
        nms_ksize (int): 非极大值抑制的核大小
        min_distance (int): 最小峰值间距
        
    Returns:
        list: 检测到的原子坐标及种类 [{"x": x, "y": y, "class": "Te"}, {"x": x, "y": y, "class": "Se"}, ...]
    """
    # 分别检测 Te 和 Se 的峰值
    te_coords = simple_find_peaks(te_heatmap, min_thresh, nms_ksize, min_distance)
    se_coords = simple_find_peaks(se_heatmap, min_thresh, nms_ksize, min_distance)
    
    # 合并两个热力图的峰值
    combined_peaks = []
    for x, y in te_coords:
        combined_peaks.append({"x": x, "y": y, "class": "Te", "value": te_heatmap[y, x]})
    for x, y in se_coords:
        combined_peaks.append({"x": x, "y": y, "class": "Se", "value": se_heatmap[y, x]})
    
    # 按热力值排序
    combined_peaks = sorted(combined_peaks, key=lambda p: p["value"], reverse=True)
    
    # 应用最小距离约束，确保同一格点内只保留热力最高的峰
    filtered_peaks = []
    for peak in combined_peaks:
        x, y = peak["x"], peak["y"]
        if all(np.sqrt((x - fp["x"])**2 + (y - fp["y"])**2) >= min_distance for fp in filtered_peaks):
            filtered_peaks.append(peak)
    
    # 返回最终的峰值列表
    return [{"x": p["x"], "y": p["y"], "class": p["class"]} for p in filtered_peaks]

def visualize_peaks(img, te_heatmap, se_heatmap, peaks, title="Peak Detection Results"):
    """
    可视化原始图像、热力图和检测到的峰值。
    
    Args:
        img (numpy.ndarray): 原始图像
        te_heatmap (numpy.ndarray): Te 原子的热力图
        se_heatmap (numpy.ndarray): Se 原子的热力图
        peaks (list): 检测到的原子坐标及种类 [{"x": x, "y": y, "class": "Te"}, {"x": x, "y": y, "class": "Se"}, ...]
        title (str): 图表标题
    """
    # 创建RGB热力图，Te为红色通道，Se为蓝色通道
    combined_heatmap = np.zeros((te_heatmap.shape[0], te_heatmap.shape[1], 3))
    combined_heatmap[:, :, 0] = te_heatmap  # Te - 红色通道
    combined_heatmap[:, :, 2] = se_heatmap  # Se - 蓝色通道
    
    # 创建带有检测结果的图像
    result_img = img.copy()
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
    
    # 在结果图像上绘制检测到的原子
    for peak in peaks:
        x, y = int(peak["x"]), int(peak["y"])
        color = (255, 0, 0) if peak["class"] == "Te" else (0, 0, 255)  # Te为红色，Se为蓝色
        cv2.circle(result_img, (x, y), 3, color, -1)
    
    # 显示图像
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(combined_heatmap)
    plt.title("Heatmap (Red=Te, Blue=Se)")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(result_img)
    plt.title("Detected Atoms")
    plt.axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show(block=True)  # 阻塞显示，等待用户关闭窗口
    plt.close()

def process_image_with_params(image_path, model_path, min_thresh=0.1, nms_ksize=5, min_distance=5, scale_factor=1.5):
    """
    使用指定的参数处理图像并可视化结果。
    
    Args:
        image_path (str): 输入图像路径
        model_path (str): 模型权重路径
        min_thresh (float): 最小阈值
        nms_ksize (int): 非极大值抑制的核大小
        min_distance (int): 最小峰值间距
        scale_factor (float): 图像放大倍数
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 初始化降噪器
    denoiser = STMDenoiser(
        filter_type='median',  # 使用中值滤波去除线噪声
        filter_size=5,         # 增大滤波核以更好地去除线噪声
        plane_subtraction=True,
        plane_order=1          # 使用平面拟合
    )
    
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}，请检查文件路径是否正确")
    
    # 放大图像
    if scale_factor != 1.0:
        height, width = img.shape
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        print(f"图像已放大 {scale_factor} 倍: {width}x{height} -> {new_width}x{new_height}")
    
    img = img.astype(np.float32) / 255.0
    
    # 对图像进行降噪处理
    denoised_img = denoiser.denoise_image(img, is_large_image=True)
    
    # 模型预测 - 使用降噪后的图像
    input_tensor = torch.from_numpy(denoised_img).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_heatmap = model(input_tensor)
        pred_heatmap = torch.sigmoid(pred_heatmap).squeeze(0).cpu().numpy()
    
    # 归一化热力图
    pred_heatmap[0] = (pred_heatmap[0] - pred_heatmap[0].min()) / (pred_heatmap[0].max() - pred_heatmap[0].min())
    pred_heatmap[1] = (pred_heatmap[1] - pred_heatmap[1].min()) / (pred_heatmap[1].max() - pred_heatmap[1].min())
    
    # 使用简化版的联合寻峰算法
    peaks = simple_find_peaks_joint(
        te_heatmap=pred_heatmap[0],
        se_heatmap=pred_heatmap[1],
        min_thresh=min_thresh,
        nms_ksize=nms_ksize,
        min_distance=min_distance
    )
    
    # 统计原子数量
    te_count = sum(1 for p in peaks if p["class"] == "Te")
    se_count = sum(1 for p in peaks if p["class"] == "Se")
    
    # 可视化结果
    title = f"Peak Detection (min_thresh={min_thresh}, nms_ksize={nms_ksize}, min_distance={min_distance})\nTe: {te_count}, Se: {se_count}"
    visualize_peaks(denoised_img, pred_heatmap[0], pred_heatmap[1], peaks, title)
    
    return peaks

def main():
    # 设置参数
    # 使用绝对路径或确保相对路径正确
    image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw", "target", "FTS0010(1).png")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "unet_final.pth")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        print("请检查文件路径是否正确，或者提供正确的图像路径")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请检查模型路径是否正确，或者提供正确的模型路径")
        return
    
    # 基本参数
    min_thresh = 0.1
    nms_ksize = 5
    min_distance = 5
    scale_factor = 1.5  # 图像放大倍数
    
    try:
        # 处理图像
        peaks = process_image_with_params(image_path, model_path, min_thresh, nms_ksize, min_distance, scale_factor)
        
        print(f"检测到 {len(peaks)} 个原子，其中 Te: {sum(1 for p in peaks if p['class'] == 'Te')}, Se: {sum(1 for p in peaks if p['class'] == 'Se')}")
    except Exception as e:
        print(f"处理图像时出错: {e}")

if __name__ == "__main__":
    main() 