import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from test import find_peaks_joint
from denoise import STMDenoiser
from scipy.spatial import Delaunay
from scipy.ndimage import maximum_filter
from tqdm import tqdm


grid_size = 14
ratio = 0.714
# 全局配置参数
CONFIG = {
    'TILE_SIZE': 512,          # 图像分割的尺寸
    'OVERLAP_THRESHOLD': int(grid_size * ratio),    # 重叠检测的阈值
    'MIN_DISTANCE': int(grid_size * ratio),         # 最小距离约束
    'GRID_SIZE': grid_size,            # 网格大小
    'MIN_THRESH': 0.01,         # 最小阈值
    'NMS_KSIZE': int(grid_size * ratio),            # 非极大值抑制的核大小
    'PEAK_MIN_DISTANCE': int(grid_size * ratio),    # 峰值检测的最小距离
    'SCALE_FACTOR': 1,          # 图像放大倍数 对于
    'RESIZE_TO': None,  # 将输入图像resize到指定的长宽 (宽, 高)，如果为None则不resize
    'OUTPUT_DIR': '../raw/target/results',  # 输出文件夹
    'INPUT_DIR': '../raw/target/images'    # 输入文件夹
}
P = []
Error= []
Te = []

from scipy.ndimage import maximum_filter, gaussian_gradient_magnitude
from skimage.feature import peak_local_max

# def determine_grid_size(heatmap, tolerance=4, min_distance=5):
#     """
#     动态确定 grid_size，通过结合局部最大值检测、非极大值抑制和梯度检测的方法。
#     Args:
#         heatmap (numpy.ndarray): 输入的热力图，形状为 (2, H, W)。
#         tolerance (int): 容错范围，允许 y 值有一定的偏移。
#         min_distance (int): 非极大值抑制的最小距离。
#     Returns:
#         int: 动态计算得到的 grid_size。
#     """
#     # 将两个通道的热力图合并为一个
#     combined_heatmap = np.maximum(heatmap[0], heatmap[1])

#     # 1. 计算梯度强度
#     gradient_magnitude = gaussian_gradient_magnitude(combined_heatmap, sigma=1)

#     # 2. 局部最大值检测
#     local_max = maximum_filter(combined_heatmap, size=3)  # 3x3 滤波器
#     peaks = (combined_heatmap == local_max)  # 局部最大值

#     # 3. 非极大值抑制
#     coordinates = peak_local_max(
#         combined_heatmap,
#         min_distance=min_distance,
#         exclude_border=False,  # 考虑边界点
#         footprint=np.ones((3, 3))  # 定义局部邻域
#     )

#     # 将非极大值抑制的结果转换为二值图像
#     nms_map = np.zeros_like(combined_heatmap, dtype=np.uint8)
#     for coord in coordinates:
#         nms_map[coord[0], coord[1]] = 1

#     # 4. 结合梯度和非极大值抑制结果
#     final_peaks = nms_map & (gradient_magnitude > 0)  # 梯度强度大于 0 的点

#     # 找到所有的非零点（可能的峰值）
#     points = np.column_stack(np.where(final_peaks > 0))  # (y, x) 坐标
#     print(f"找到 {len(points)} 个可能的峰值点")
#     print(f"热力图的形状: {heatmap.shape}")

#     # 按 y 值分组，统计每一行的点数
#     rows = {}
#     for y, x in points:
#         found_row = False
#         for row_y in rows.keys():
#             if abs(row_y - y) <= tolerance:  # 容错范围内归为同一行
#                 rows[row_y].append(x)
#                 found_row = True
#                 break
#         if not found_row:
#             rows[y] = [x]

#     # 找到点数最多的一行
#     max_row = max(rows.values(), key=len)
#     print(f"最多的行有 {len(max_row)} 个点")

#     # 计算 grid_size
#     image_width = heatmap.shape[2]  # 热力图的宽度
#     grid_size = int(image_width / len(max_row))
#     print(f"动态计算的 grid_size: {grid_size}")
#     return grid_size

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
    # tiles = []
    # for i in range(0, height, tile_size):
    #     for j in range(0, width, tile_size):
    #         tile = image[i:i + tile_size, j:j + tile_size]
    #         tiles.append((tile, (i, j)))
    # return tiles
    # 计算能被整除的区域
    valid_height = (height // tile_size) * tile_size
    valid_width = (width // tile_size) * tile_size

    tiles = []
    for i in range(0, valid_height, tile_size):
        for j in range(0, valid_width, tile_size):
            tile = image[i:i + tile_size, j:j + tile_size]
            tiles.append((tile, (i, j)))
    
    return tiles

def infer_and_detect(image, model, device, tile_size=None, original_img=None, filename=None):
    """
    对输入图像进行推理和寻峰，并保存整张图像的降噪图、热力图和标注结果。
    """
    if tile_size is None:
        tile_size = CONFIG['TILE_SIZE']
    
    tiles = split_image(image, tile_size)
    original_tiles = split_image(original_img, tile_size)  # 分割原始图像
    height, width = image.shape
    combined_heatmap = np.zeros((2, height, width))  # 用于存储整张图像的热力图
    denoised_image = np.zeros_like(image)  # 用于存储整张图像的降噪结果
    all_coords = []

    for (tile, (x, y)), (original_tile, _) in tqdm(zip(tiles, original_tiles), total=len(tiles), desc="Processing tiles"):
        # 转换为张量并推理
        tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_heatmap = model(tile_tensor).squeeze(0).cpu().numpy()
        
        # 归一化热力图
        pred_heatmap[0] = (pred_heatmap[0] - pred_heatmap[0].min()) / (pred_heatmap[0].max() - pred_heatmap[0].min())
        pred_heatmap[1] = (pred_heatmap[1] - pred_heatmap[1].min()) / (pred_heatmap[1].max() - pred_heatmap[1].min())
        
        # 合并热力图
        combined_heatmap[0, x:x+tile.shape[0], y:y+tile.shape[1]] = pred_heatmap[0]
        combined_heatmap[1, x:x+tile.shape[0], y:y+tile.shape[1]] = pred_heatmap[1]
        
        # 合并降噪图像
        denoised_image[x:x+tile.shape[0], y:y+tile.shape[1]] = tile
        
        # if CONFIG['GRID_SIZE'] == 0:
        #     # 动态确定 grid_size
        #     CONFIG['GRID_SIZE'] = determine_grid_size(pred_heatmap)
        #     print(f"动态确定的 grid_size: {CONFIG['GRID_SIZE']}")

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
    
    # 使用 Delaunay 三角剖分区分边界点和内部点
    center_coords, border_coords = delaunay_boundary_detection(all_coords)

    # 统计内部和边界原子的数量
    center_te_count = sum(1 for coord in center_coords if coord["class"] == "Te")
    center_se_count = sum(1 for coord in center_coords if coord["class"] == "Se")
    border_te_count = sum(1 for coord in border_coords if coord["class"] == "Te")
    border_se_count = sum(1 for coord in border_coords if coord["class"] == "Se")

    # 计算 Te 的掺杂比例
    center_total_atoms = center_te_count + center_se_count
    border_total_atoms = border_te_count + border_se_count
    center_te_ratio = (center_te_count / center_total_atoms) * 100 if center_total_atoms > 0 else 0
    border_te_ratio = (border_te_count / border_total_atoms) * 100 if border_total_atoms > 0 else 0

    pc = center_te_ratio/100
    po = border_te_ratio/100
    # SE = np.sqrt(po*(1-po)/border_total_atoms)
    # dp = np.abs(pc-po)
    # error = np.sqrt(SE**2 + dp**2)*100
    error = 1
    # dp = po-pc
    # pc = pc+1/2*dp
    # error = 1/2*np.abs(dp)*100

    P.append(pc*100)
    Error.append(error)
    Te.append(center_te_count+border_te_count)

    # 输出到文件
    output_file = os.path.join(CONFIG['OUTPUT_DIR'], "results.txt")
    with open(output_file, "a") as f:
        f.write(f"File: {filename}, Center - Te: {center_te_count}, Se: {center_se_count}, Te Ratio: {center_te_ratio:.2f}%, "
                f"Border - Te: {border_te_count}, Se: {border_se_count}, Te Ratio: {border_te_ratio:.2f}%\n")
        print(f"结果已保存到: {output_file}")

    # 在原图上绘制标注
    annotated_image = (original_img * 255).astype(np.uint8)  # 将原图转换为 uint8
    if len(annotated_image.shape) == 2:  # 如果是灰度图，转换为 RGB
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_GRAY2RGB)
    
    # 绘制内部原子
    for coord in center_coords:
        if coord["class"] == "Te":
            color = (0, 0, 255)  # 红色
        else:
            color = (0, 255, 0)  # 绿色
        cv2.circle(annotated_image, (coord["x"], coord["y"]), 3, color, -1)
    
    # 绘制边界原子
    for coord in border_coords:
        if coord["class"] == "Te":
            color = (0, 255, 255)  # 黄色
        else:
            color = (255, 0, 0)  # 蓝色
        cv2.circle(annotated_image, (coord["x"], coord["y"]), 3, color, -1)

    # 保存标注图像
    annotated_path = os.path.join(CONFIG['OUTPUT_DIR'], f"{filename}_annotated.png")
    cv2.imwrite(annotated_path, annotated_image)
    print(f"标注图像已保存到: {annotated_path}")

    # 保存降噪图像
    denoised_path = os.path.join(CONFIG['OUTPUT_DIR'], f"{filename}_denoised.png")
    cv2.imwrite(denoised_path, (denoised_image * 255).astype(np.uint8))
    print(f"降噪图像已保存到: {denoised_path}")

    # 保存热力图
    heatmap_path = os.path.join(CONFIG['OUTPUT_DIR'], f"{filename}_heatmap.png")
    combined_heatmap_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    combined_heatmap_rgb[:, :, 0] = (combined_heatmap[0] * 255).astype(np.uint8)  # Te - 红色通道
    combined_heatmap_rgb[:, :, 2] = (combined_heatmap[1] * 255).astype(np.uint8)  # Se - 蓝色通道
    cv2.imwrite(heatmap_path, combined_heatmap_rgb)
    print(f"热力图已保存到: {heatmap_path}")

    return all_coords

def show_tile_results(original_tile, tile, pred_heatmap, coords, x, y, tile_size, filename):
    """
    显示并保存当前处理的图像、热力图和识别结果。
    Args:
        original_tile (numpy.ndarray): 原始图像块。
        tile (numpy.ndarray): 降噪后的图像块。
        pred_heatmap (numpy.ndarray): 预测的热力图。
        coords (list): 检测到的原子坐标。
        x (int): 图像块在原始图像中的x坐标。
        y (int): 图像块在原始图像中的y坐标。
        tile_size (int): 图像块的尺寸。
        filename (str): 当前处理的文件名。
    """
    # 创建RGB热力图，Te为红色通道，Se为蓝色通道
    combined_heatmap = np.zeros((pred_heatmap.shape[1], pred_heatmap.shape[2], 3))
    combined_heatmap[:, :, 0] = pred_heatmap[0]  # Te - 红色通道
    combined_heatmap[:, :, 2] = pred_heatmap[1]  # Se - 蓝色通道
    
    # 创建带有检测结果的图像
    result_img = tile.copy()
    result_img = (result_img * 255).astype(np.uint8)  # 将浮点数转换为 0-255 的 uint8
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
    
    # 在结果图像上绘制检测到的原子
    te_count = 0
    se_count = 0
    for coord in coords:
        rel_x = coord["x"] - y
        rel_y = coord["y"] - x
        if 0 <= rel_x < tile_size and 0 <= rel_y < tile_size:
            color = (255, 0, 0) if coord["class"] == "Te" else (0, 0, 255)
            cv2.circle(result_img, (rel_x, rel_y), 3, color, -1)
            if coord["class"] == "Te":
                te_count += 1
            else:
                se_count += 1
    
    # 计算 Te 的掺杂比例
    total_atoms = te_count + se_count
    te_ratio = (te_count / total_atoms) * 100 if total_atoms > 0 else 0

    # 在图像上打印文件名、Te 和 Se 数量以及 Te 的掺杂比例
    text = f"File: {filename}, Te: {te_count}, Se: {se_count}, Te Ratio: {te_ratio:.2f}%"
    cv2.putText(result_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 保存结果图像
    output_dir = CONFIG['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}_result.png")
    cv2.imwrite(output_path, result_img)
    print(f"结果图像已保存到: {output_path}")

    # # 显示图像
    # plt.figure(figsize=(20, 5))
    # plt.subplot(1, 4, 1)
    # plt.imshow(original_tile, cmap="gray")
    # plt.title("Original Tile")
    # plt.axis("off")
    
    # plt.subplot(1, 4, 2)
    # plt.imshow(tile, cmap="gray")
    # plt.title("Denoised Tile")
    # plt.axis("off")
    
    # plt.subplot(1, 4, 3)
    # plt.imshow(combined_heatmap)
    # plt.title("Heatmap")
    # plt.axis("off")
    
    # plt.subplot(1, 4, 4)
    # plt.imshow(result_img)
    # plt.title("Detected Atoms")
    # plt.axis("off")
    
    # plt.tight_layout()
    # # plt.show()
    # plt.close()

def process_image(image_path, model, device, tile_size=None):
    """
    对单张图像进行处理，生成热力图和坐标文件。
    """
    if tile_size is None:
        tile_size = CONFIG['TILE_SIZE']
    
    # 读取图像
    filename = os.path.basename(image_path).split('.')[0]
    print(f"处理图像: {filename}")
    original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0  # 原始图像
    img = original_img.copy()  # 复制一份用于降噪处理

    # 可选：将图像resize到指定大小
    if CONFIG['RESIZE_TO'] is not None:
        resize_width, resize_height = CONFIG['RESIZE_TO']
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        original_img = cv2.resize(original_img, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)
        print(f"图像已resize到: {resize_width}x{resize_height}")
    
    # 对图像进行降噪处理
    denoiser = STMDenoiser(
        filter_type='median',  # 使用中值滤波去除线噪声
        filter_size=5,         # 增大滤波核以更好地去除线噪声
        plane_subtraction=True,
        plane_order=1          # 使用平面拟合
    )
    img = denoiser.denoise_image(img, is_large_image=True)  # 降噪后的图像
    
    # 放大图像
    if CONFIG['SCALE_FACTOR'] != 1.0:
        height, width = img.shape
        new_height = int(height * CONFIG['SCALE_FACTOR'])
        new_width = int(width * CONFIG['SCALE_FACTOR'])
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        original_img = cv2.resize(original_img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        print(f"图像已放大 {CONFIG['SCALE_FACTOR']} 倍: {width}x{height} -> {new_width}x{new_height}")
    
    # 推理和寻峰
    all_coords = infer_and_detect(img, model, device, tile_size, original_img, filename)

def delaunay_boundary_detection(coords, image_shape=None, margin=50):
    """
    基于 Delaunay 三角剖分区分边界点和内部点，并考虑图像边缘的点。
    Args:
        coords (list): 原子坐标列表，每个坐标是一个字典，包含 "x" 和 "y"。
        image_shape (tuple): 图像的形状 (height, width)，用于检测图像边缘的点。
        margin (int): 边缘点的检测范围。
    Returns:
        tuple: (内部点列表, 边界点列表)
    """
    # 提取点的坐标
    points = np.array([[coord["x"], coord["y"]] for coord in coords])
    
    # 进行 Delaunay 三角剖分
    tri = Delaunay(points)
    
    # 统计边的出现次数
    edge_count = {}
    for simplex in tri.simplices:
        edges = [
            tuple(sorted([simplex[0], simplex[1]])),
            tuple(sorted([simplex[1], simplex[2]])),
            tuple(sorted([simplex[2], simplex[0]]))
        ]
        for edge in edges:
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1
    
    # 提取边界边（出现次数为 1 的边）
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    # 提取边界点
    boundary_points = set()
    for edge in boundary_edges:
        boundary_points.update(edge)
    
    # 如果提供了图像形状，进一步检测图像边缘的点
    if image_shape is not None:
        height, width = image_shape
        for i, point in enumerate(points):
            if (
                point[0] < margin or point[0] > width - margin or
                point[1] < margin or point[1] > height - margin
            ):
                boundary_points.add(i)
    
    # 提取内部点
    all_points = set(range(len(points)))
    internal_points = all_points - boundary_points
    
    # 将点索引转换回原始坐标
    boundary_coords = [coords[i] for i in boundary_points]
    internal_coords = [coords[i] for i in internal_points]
    
    return internal_coords, boundary_coords

import matplotlib.pyplot as plt

def plot_doping_concentration(output_dir):
    """
    根据结果文件绘制掺杂浓度曲线。
    Args:
        results_file (str): 包含掺杂浓度结果的文件路径。
        output_dir (str): 保存绘图的输出目录。
    """
    # 读取结果文件
    image_indices = np.arange(1,21)
    center_ratios = P
    errors = Error

    # 绘制掺杂浓度曲线
    plt.figure(figsize=(10, 6))
    plt.errorbar(image_indices, center_ratios, yerr=errors, fmt='-o', capsize=5, label="Te Doping Concentration")
    plt.plot(image_indices, Te, 'r--', label="Total Te Atoms")
    plt.xlabel("Image Index")
    plt.ylim(580,760)
    plt.ylabel("Te Doping Concentration (%)")
    plt.title("Te Doping Concentration vs Image Index")
    plt.grid(True)
    plt.legend()

    # 保存图像
    plot_path = os.path.join(output_dir, "doping_concentration_plot.png")
    plt.savefig(plot_path)
    print(f"掺杂浓度曲线已保存到: {plot_path}")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    # model.load_state_dict(torch.load("checkpoints/unet_final.pth", map_location=device))
    model.load_state_dict(torch.load("../archive/250413-augmented-1000epoch/unet_final.pth", map_location=device))
    model.eval()

    input_dir = CONFIG['INPUT_DIR']
    output_file = os.path.join(CONFIG['OUTPUT_DIR'], "results.txt")
    
    # 清空结果文件
    with open(output_file, "w") as f:
        f.write("")  # 清空文件内容
    
    # 获取文件列表并按序号排序
    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(x.split('-')[0])  # 提取文件名中的序号并排序
    )
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"处理图像: {image_path}")
        process_image(image_path, model, device)
    
    plot_doping_concentration(CONFIG['OUTPUT_DIR'])

if __name__ == "__main__":
    main()