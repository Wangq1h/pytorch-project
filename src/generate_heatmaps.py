import os
import json
import numpy as np
import cv2

def generate_heatmap(coords, img_shape, sigma=2, intensity=1.0):
    """
    根据坐标生成高斯热力图。
    Args:
        coords (list): 原子坐标列表 [(x1, y1), (x2, y2), ...]。
        img_shape (tuple): 图像尺寸 (height, width)。
        sigma (int): 高斯核的标准差。
        intensity (float): 热力图峰值的强度。
    Returns:
        np.ndarray: 生成的热力图。
    """
    heatmap = np.zeros(img_shape, dtype=np.float32)
    for (x, y) in coords:
        if 0 <= x < img_shape[1] and 0 <= y < img_shape[0]:
            heatmap[y, x] = intensity  # 在指定坐标处设置为指定强度
    # 应用高斯模糊
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigma)
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)  # 如果热力图全为 0，则直接设置为全 0
    return heatmap

def main():
    # img_dir = "../data/images"  # 原始图像文件夹
    # label_dir = "../data/labels"  # JSON 文件夹
    # output_dir = "../data/labels_npy"  # 输出 .npy 文件夹
    img_dir = "../raw/target"  # 原始图像文件夹
    label_dir = "../raw/target"  # JSON 文件夹
    output_dir = "../raw/target"  # 输出 .npy 文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 labels 文件夹中的每个 JSON 文件
    for json_file in os.listdir(label_dir):
        if not json_file.endswith(".json"):
            continue

        # 加载 JSON 文件
        json_path = os.path.join(label_dir, json_file)
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        # 获取对应的图像文件路径
        img_name = json_file.replace('.json', '.png')  # 假设图像文件是 .png 格式
        img_path = os.path.join(img_dir, img_name)

        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"Warning: Image file {img_name} not found. Skipping...")
            continue

        # 读取图像以获取尺寸
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Failed to load image {img_name}. Skipping...")
            continue
        img_shape = img.shape

        # 准备热力图
        te_coords = [(atom['x'], atom['y']) for atom in annotations if atom['class'] == 'Te']
        se_coords = [(atom['x'], atom['y']) for atom in annotations if atom['class'] == 'Se']

        te_heatmap = generate_heatmap(te_coords, img_shape, sigma=2, intensity=1.0)  # Te 原子强度为 1.0
        se_heatmap = generate_heatmap(se_coords, img_shape, sigma=2, intensity=1.2)  # Se 原子强度为 1.2

        # 堆叠热力图并保存为 .npy 文件
        heatmap = np.stack((te_heatmap, se_heatmap), axis=0)  # shape: (2, height, width)
        npy_file = os.path.join(output_dir, json_file.replace('.json', '.npy'))
        np.save(npy_file, heatmap)

        print(f"Generated heatmap for {img_name} -> {npy_file}")

if __name__ == "__main__":
    main()