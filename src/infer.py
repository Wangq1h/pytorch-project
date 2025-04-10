import os
import cv2
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from models.unet import UNet
from test import find_peaks_joint
from denoise import STMDenoiser
import time
import cProfile
import pstats
from pstats import SortKey

# 全局配置参数
CONFIG = {
    'TILE_SIZE': 1024,          # 图像分割的尺寸
    'OVERLAP_THRESHOLD': 8,    # 重叠检测的阈值
    'MIN_DISTANCE': 8,         # 最小距离约束
    'GRID_SIZE': 12,           # 网格大小
    'MIN_THRESH': 0.01,         # 最小阈值
    'NMS_KSIZE': 8,            # 非极大值抑制的核大小
    'PEAK_MIN_DISTANCE': 8,    # 峰值检测的最小距离
    'SCALE_FACTOR': 1.5        # 图像放大倍数
}

def profile_function(func):
    """性能分析装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

@profile_function
def split_image(image, tile_size=None):
    """
    将图像切割成 tile_size x tile_size 的小图片。
    Args:
        image (numpy.ndarray): 输入图像。
        tile_size (int): 小图片的尺寸，如果为None则使用全局配置。
    Returns:
        list: 切割后的图片列表及其对应的坐标 [(tile, (x, y)), ...]。
    """
    if tile_size is None:
        tile_size = CONFIG['TILE_SIZE']
    height, width = image.shape
    tiles = []
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = image[i:i + tile_size, j:j + tile_size]
            tiles.append((tile, (i, j)))
    return tiles

@profile_function
def merge_heatmaps(heatmaps, original_size, tile_size=None):
    """
    将小图片的热力图拼合成完整的热力图。
    Args:
        heatmaps (list): 小图片的热力图及其对应的坐标 [(heatmap, (x, y)), ...]。
        original_size (tuple): 原始图像的尺寸 (height, width)。
        tile_size (int): 小图片的尺寸，如果为None则使用全局配置。
    Returns:
        numpy.ndarray: 拼合后的完整热力图。
    """
    if tile_size is None:
        tile_size = CONFIG['TILE_SIZE']
    full_heatmap = np.zeros((2, original_size[0], original_size[1]), dtype=np.float32)
    for heatmap, (x, y) in heatmaps:
        full_heatmap[:, x:x + tile_size, y:y + tile_size] = heatmap
    return full_heatmap

@profile_function
def infer_and_detect(image, model, device, tile_size=None, overlap_threshold=None):
    """
    对输入图像进行推理和寻峰，并去除拼合边界上的重复检测。
    Args:
        image (numpy.ndarray): 输入图像。
        model (torch.nn.Module): 训练好的模型。
        device (torch.device): 运行设备。
        tile_size (int): 小图片的尺寸，如果为None则使用全局配置。
        overlap_threshold (int): 去重时的距离阈值，如果为None则使用全局配置。
    Returns:
        numpy.ndarray: 拼合后的完整热力图。
        list: 检测到的原子坐标 [{"x": x, "y": y, "class": "Te"}, ...]。
    """
    if tile_size is None:
        tile_size = CONFIG['TILE_SIZE']
    if overlap_threshold is None:
        overlap_threshold = CONFIG['OVERLAP_THRESHOLD']
    
    # 增加批处理大小
    batch_size = 64  # 增加批处理大小以提高GPU利用率
    
    tiles = split_image(image, tile_size)
    heatmaps = []
    all_coords = []
    
    # 批量处理
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i:i + batch_size]
        # 使用torch.stack一次性创建批次张量
        batch_tensors = torch.stack([torch.from_numpy(tile[0]).unsqueeze(0).float() for tile in batch_tiles]).to(device)
        
        with torch.no_grad():
            batch_pred_heatmaps = model(batch_tensors)
        
        # 将结果移到CPU并转换为numpy数组
        batch_pred_heatmaps = batch_pred_heatmaps.cpu().numpy()
        
        for idx, (tile, (x, y)) in enumerate(batch_tiles):
            # 直接使用numpy数组，不需要squeeze
            pred_heatmap = batch_pred_heatmaps[idx]
            
            # 归一化热力图
            pred_heatmap[0] = (pred_heatmap[0] - pred_heatmap[0].min()) / (pred_heatmap[0].max() - pred_heatmap[0].min())
            pred_heatmap[1] = (pred_heatmap[1] - pred_heatmap[1].min()) / (pred_heatmap[1].max() - pred_heatmap[1].min())
            heatmaps.append((pred_heatmap, (x, y)))

            # 寻峰
            coords = find_peaks_joint(
                te_heatmap=pred_heatmap[0],
                se_heatmap=pred_heatmap[1],
                grid_size=CONFIG['GRID_SIZE'],
                min_thresh=CONFIG['MIN_THRESH'],
                nms_ksize=CONFIG['NMS_KSIZE'],
                min_distance=CONFIG['PEAK_MIN_DISTANCE']
            )
            # 调整坐标到全图范围
            for coord in coords:
                coord["x"] += y
                coord["y"] += x
            all_coords.extend(coords)
            
            # 注释掉显示当前处理的图像、热力图和识别结果
            # show_tile_results(tile, pred_heatmap, coords, x, y, tile_size)

    # 使用分块计算去除重复检测的原子
    if all_coords:
        coords_array = np.array([[c["x"], c["y"]] for c in all_coords])
        n_points = len(all_coords)
        
        # 创建掩码，初始全部保留
        mask = np.ones(n_points, dtype=bool)
        
        # 分块大小，可以根据可用内存调整
        block_size = 1000
        
        # 分块计算距离
        for i in range(0, n_points, block_size):
            end_i = min(i + block_size, n_points)
            # 计算当前块与所有点的距离
            for j in range(i, end_i):
                # 计算当前点与所有其他点的距离
                point_coords = coords_array[j]
                other_coords = coords_array
                
                # 计算距离
                diff = point_coords - other_coords
                distances = np.sqrt(np.sum(diff * diff, axis=1))
                
                # 排除自身
                distances[j] = np.inf
                
                # 如果与任何点的距离小于重叠阈值，则移除当前点
                if np.any(distances < overlap_threshold):
                    mask[j] = False
        
        unique_coords = [c for i, c in enumerate(all_coords) if mask[i]]
    else:
        unique_coords = []

    # 拼合热力图
    full_heatmap = merge_heatmaps(heatmaps, image.shape)
    return full_heatmap, unique_coords

def show_tile_results(tile, pred_heatmap, coords, x, y, tile_size):
    """
    显示当前处理的图像、热力图和识别结果。
    Args:
        tile (numpy.ndarray): 当前处理的图像块。
        pred_heatmap (numpy.ndarray): 预测的热力图。
        coords (list): 检测到的原子坐标。
        x (int): 图像块在原始图像中的x坐标。
        y (int): 图像块在原始图像中的y坐标。
        tile_size (int): 图像块的尺寸。
    """
    # 创建RGB热力图，Te为红色通道，Se为蓝色通道
    combined_heatmap = np.zeros((pred_heatmap.shape[1], pred_heatmap.shape[2], 3))
    combined_heatmap[:, :, 0] = pred_heatmap[0]  # Te - 红色通道
    combined_heatmap[:, :, 2] = pred_heatmap[1]  # Se - 蓝色通道
    
    # 创建带有检测结果的图像
    result_img = tile.copy()
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
    
    # 在结果图像上绘制检测到的原子
    for coord in coords:
        # 将坐标转换为图像块内的相对坐标
        rel_x = coord["x"] - x
        rel_y = coord["y"] - y
        
        # 确保坐标在图像块内
        if 0 <= rel_x < tile_size and 0 <= rel_y < tile_size:
            color = (255, 0, 0) if coord["class"] == "Te" else (0, 0, 255)  # Te为红色，Se为蓝色
            cv2.circle(result_img, (rel_x, rel_y), 3, color, -1)
    
    # 注释掉显示图像的代码
    # plt.figure(figsize=(15, 5))
    
    # plt.subplot(1, 3, 1)
    # plt.imshow(tile, cmap="gray")
    # plt.title("Original Tile")
    # plt.axis("off")
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(combined_heatmap)
    # plt.title("Heatmap")
    # plt.axis("off")
    
    # plt.subplot(1, 3, 3)
    # plt.imshow(result_img)
    # plt.title("Detected Atoms")
    # plt.axis("off")
    
    # plt.tight_layout()
    # plt.show(block=True)  # 阻塞显示，等待用户关闭窗口
    # plt.close()

import matplotlib.patches as mpatches

def save_visualization(image, full_heatmap, all_coords, te_count, se_count, ratio, output_path):
    # 减少matplotlib的自动缩放操作
    plt.ioff()  # 关闭交互模式
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 预先计算显示范围
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    
    # 使用更高效的方式绘制散点
    te_coords = np.array([[c["x"], c["y"]] for c in all_coords if c["class"] == "Te"])
    se_coords = np.array([[c["x"], c["y"]] for c in all_coords if c["class"] == "Se"])
    
    if len(te_coords) > 0:
        ax.scatter(te_coords[:, 0], te_coords[:, 1], c="red", s=10, alpha=0.8, edgecolors="black", linewidths=0.5)
    if len(se_coords) > 0:
        ax.scatter(se_coords[:, 0], se_coords[:, 1], c="blue", s=10, alpha=0.8, edgecolors="black", linewidths=0.5)
    
    # 添加图例
    te_patch = mpatches.Patch(color="red", label=f"Te Atoms: {te_count}")
    se_patch = mpatches.Patch(color="blue", label=f"Se Atoms: {se_count}")
    ax.legend(handles=[te_patch, se_patch], loc="upper right", fontsize=10)

    # 添加标题
    ax.set_title(f"Visualization (Te/Se Ratio: {ratio:.2f})", fontsize=14)
    ax.axis("off")

    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

@profile_function
def apply_min_distance_constraint(coords, min_distance=None):
    """
    对检测到的原子坐标应用最小距离约束，确保每个格点中只保留一个峰值。
    使用分块计算来减少内存使用。
    """
    if min_distance is None:
        min_distance = CONFIG['MIN_DISTANCE']
    
    if not coords:  # 如果坐标列表为空，直接返回
        return coords
    
    # 使用numpy向量化操作替代循环
    coords_array = np.array([[c["x"], c["y"]] for c in coords])
    n_points = len(coords)
    
    # 创建掩码，初始全部保留
    mask = np.ones(n_points, dtype=bool)
    
    # 分块大小，可以根据可用内存调整
    block_size = 1000
    
    # 分块计算距离
    for i in range(0, n_points, block_size):
        end_i = min(i + block_size, n_points)
        # 计算当前块与所有点的距离
        for j in range(i, end_i):
            # 计算当前点与所有其他点的距离
            point_coords = coords_array[j]
            other_coords = coords_array
            
            # 计算距离
            diff = point_coords - other_coords
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            
            # 排除自身
            distances[j] = np.inf
            
            # 如果与任何点的距离小于最小距离，则移除当前点
            if np.any(distances < min_distance):
                mask[j] = False
    
    filtered_coords = [c for i, c in enumerate(coords) if mask[i]]
    
    return filtered_coords

@profile_function
def process_image(image_path, model, device, output_dir, tile_size=None):
    """
    对单张图像进行处理，生成热力图和坐标文件，并保存可视化图片。
    """
    if tile_size is None:
        tile_size = CONFIG['TILE_SIZE']
    
    # 使用生成器而不是一次性加载所有数据
    def image_generator():
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        # 对图像进行降噪处理
        denoiser = STMDenoiser(
            filter_type='median',  # 使用中值滤波去除线噪声
            filter_size=5,         # 增大滤波核以更好地去除线噪声
            plane_subtraction=True,
            plane_order=1          # 使用平面拟合
        )
        img = denoiser.denoise_image(img, is_large_image=True)
        
        # 放大图像
        if CONFIG['SCALE_FACTOR'] != 1.0:
            height, width = img.shape
            new_height = int(height * CONFIG['SCALE_FACTOR'])
            new_width = int(width * CONFIG['SCALE_FACTOR'])
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            print(f"图像已放大 {CONFIG['SCALE_FACTOR']} 倍: {width}x{height} -> {new_width}x{new_height}")
        
        height, width = img.shape
        new_height = (height + tile_size - 1) // tile_size * tile_size
        new_width = (width + tile_size - 1) // tile_size * tile_size
        padded_img = np.zeros((new_height, new_width), dtype=np.float32)
        padded_img[:height, :width] = img
        return padded_img

    # 推理和寻峰
    padded_img = image_generator()
    full_heatmap, all_coords = infer_and_detect(padded_img, model, device, tile_size)

    # 应用最小距离约束
    all_coords = apply_min_distance_constraint(all_coords, min_distance=10)

    # 保存热力图
    heatmap_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_heatmap.npy")
    np.save(heatmap_path, full_heatmap)
    print(f"Saved heatmap to {heatmap_path}")

    # 保存坐标文件，使用更高效的JSON序列化
    coords_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_coords.json")
    # 将坐标转换为简单的列表格式
    coords_list = [[int(coord["x"]), int(coord["y"]), coord["class"]] for coord in all_coords]
    with open(coords_path, "w") as f:
        json.dump(coords_list, f, separators=(',', ':'))  # 使用更紧凑的格式
    print(f"Saved coordinates to {coords_path}")

    # 统计原子数目和比值
    te_count = sum(1 for coord in all_coords if coord["class"] == "Te")
    se_count = sum(1 for coord in all_coords if coord["class"] == "Se")
    ratio = te_count / se_count if se_count > 0 else float('inf')
    print(f"Te atoms: {te_count}, Se atoms: {se_count}, Te/Se ratio: {ratio:.2f}")

    # 保存可视化图片
    vis_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + "_visualization.png")
    save_visualization(padded_img, full_heatmap, all_coords, te_count, se_count, ratio, vis_path)

def main():
    # 创建性能分析器
    profiler = cProfile.Profile()
    profiler.enable()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load("models/unet_final.pth", map_location=device))
    model.eval()

    input_dir = "../raw/target"
    output_dir = "../raw/target/results"
    os.makedirs(output_dir, exist_ok=True)

    for image_file in os.listdir(input_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_file)
            process_image(image_path, model, device, output_dir)

    # 停止性能分析器并打印结果
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats(SortKey.TIME)
    stats.print_stats(20)  # 打印前20个最耗时的函数

if __name__ == "__main__":
    main()