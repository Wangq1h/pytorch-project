import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from test import find_peaks_joint
from denoise import STMDenoiser

# 全局配置参数
CONFIG = {
    'TILE_SIZE': 1024,          # 图像分割的尺寸
    'OVERLAP_THRESHOLD': 10,    # 重叠检测的阈值
    'MIN_DISTANCE': 10,         # 最小距离约束
    'GRID_SIZE': 14,           # 网格大小
    'MIN_THRESH': 0.01,         # 最小阈值
    'NMS_KSIZE': 10,            # 非极大值抑制的核大小
    'PEAK_MIN_DISTANCE': 10,    # 峰值检测的最小距离
    'SCALE_FACTOR': 1,         # 图像放大倍数
    'OUTPUT_DIR': '../raw/target/results',  # 输出文件夹
    'INPUT_DIR': '../raw/target/images'  # 输入文件夹
}

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

    for (tile, (x, y)), (original_tile, _) in zip(tiles, original_tiles):
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
    
    # 绘制整张图像的标注结果
    annotated_image = cv2.cvtColor((denoised_image * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    te_count = 0
    se_count = 0
    for coord in all_coords:
        color = (255, 0, 0) if coord["class"] == "Se" else (0, 0, 255)
        cv2.circle(annotated_image, (coord["x"], coord["y"]), 3, color, -1)
        if coord["class"] == "Te":
            te_count += 1
        else:
            se_count += 1

    # 计算 Te 的掺杂比例
    total_atoms = te_count + se_count
    te_ratio = (te_count / total_atoms) * 100 if total_atoms > 0 else 0

    # 在标注图像上打印信息
    text = f"File: {filename}, Te: {te_count}, Se: {se_count}, Te Ratio: {te_ratio:.2f}%"
    print(text)
    # # 扩展图像高度以容纳文本
    # text_height = 50  # 文本区域高度
    # annotated_with_text = np.zeros((annotated_image.shape[0] + text_height, annotated_image.shape[1], 3), dtype=np.uint8)
    # annotated_with_text[:annotated_image.shape[0], :, :] = annotated_image  # 将原始图像复制到扩展图像中

    # 在扩展区域中打印文本
    # cv2.putText(annotated_with_text, text, (10, annotated_image.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 保存整张图像的结果
    output_dir = CONFIG['OUTPUT_DIR']
    os.makedirs(output_dir, exist_ok=True)

    # 保存降噪图像
    denoised_path = os.path.join(output_dir, f"{filename}_denoised.png")
    cv2.imwrite(denoised_path, (denoised_image * 255).astype(np.uint8))
    print(f"降噪图像已保存到: {denoised_path}")

    # 保存热力图
    heatmap_path = os.path.join(output_dir, f"{filename}_heatmap.png")
    combined_heatmap_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    combined_heatmap_rgb[:, :, 0] = (combined_heatmap[0] * 255).astype(np.uint8)  # Te - 红色通道
    combined_heatmap_rgb[:, :, 2] = (combined_heatmap[1] * 255).astype(np.uint8)  # Se - 蓝色通道
    cv2.imwrite(heatmap_path, combined_heatmap_rgb)
    print(f"热力图已保存到: {heatmap_path}")

    # 保存标注图像
    annotated_path = os.path.join(output_dir, f"{filename}_annotated.png")
    cv2.imwrite(annotated_path, annotated_image)
    print(f"标注图像已保存到: {annotated_path}")

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load("checkpoints/unet_final.pth", map_location=device))
    model.eval()

    input_dir = CONFIG['INPUT_DIR']
    for image_file in os.listdir(input_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_file)
            print(f"处理图像: {image_path}")
            process_image(image_path, model, device)

if __name__ == "__main__":
    main()