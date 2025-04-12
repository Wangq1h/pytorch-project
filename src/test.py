import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from dataset import FeTeSeDataset
from denoise import STMDenoiser

def find_peaks(heatmap, min_thresh=0.1, nms_ksize=5, min_distance=10):
    """
    从热力图中检测峰值。
    Args:
        heatmap (numpy.ndarray): 输入热力图。
        min_thresh (float): 最小阈值，用于过滤低强度伪峰值。
        nms_ksize (int): 非极大值抑制的核大小。
        min_distance (int): 最小峰值间距。
    Returns:
        list: 检测到的峰值坐标 [(x1, y1), (x2, y2), ...]。
    """
    # 检查热力图是否全为 0
    if heatmap.max() == 0:
        return []  # 返回空列表，表示没有峰值

    # 动态阈值计算
    dynamic_thresh = (heatmap.max() - heatmap.min()) / 10 + heatmap.min()
    thresh = max(dynamic_thresh, min_thresh)  # 使用动态阈值和最小阈值的较大值

    # 创建二值掩码
    mask = (heatmap > thresh).astype(np.uint8)

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

def find_peaks_joint(te_heatmap, se_heatmap, grid_size=10, min_thresh=0.1, nms_ksize=5, min_distance=8):
    """
    从两个热力图中联合检测峰值，确保同一格点内只保留热力最高的峰。
    Args:
        te_heatmap (numpy.ndarray): Te 原子的热力图。
        se_heatmap (numpy.ndarray): Se 原子的热力图。
        grid_size (int): 原子阵列的格点大小（像素）。
        min_thresh (float): 最小阈值，用于过滤低强度伪峰值。
        nms_ksize (int): 非极大值抑制的核大小。
        min_distance (int): 最小峰值间距。
    Returns:
        list: 检测到的原子坐标及种类 [{"x": x, "y": y, "class": "Te"}, {"x": x, "y": y, "class": "Se"}, ...]。
    """
    se_weight = 2
    def detect_peaks(heatmap, thresh):
        """辅助函数：检测单个热力图中的峰值。"""
        mask = (heatmap > thresh).astype(np.uint8)
        max_map = cv2.dilate(heatmap, np.ones((nms_ksize, nms_ksize), np.uint8))
        peak_mask = (heatmap == max_map) & (mask == 1)
        peak_y, peak_x = np.where(peak_mask)
        coords = list(zip(peak_x, peak_y))
        return sorted(coords, key=lambda c: heatmap[c[1], c[0]], reverse=True)

    # 初始阈值检测
    te_coords = detect_peaks(te_heatmap, min_thresh)
    se_coords = detect_peaks(se_heatmap, min_thresh)

    # 合并两个热力图的峰值
    combined_peaks = []
    for x, y in te_coords:
        combined_peaks.append({"x": x, "y": y, "class": "Te", "value": te_heatmap[y, x]})
    for x, y in se_coords:
        combined_peaks.append({"x": x, "y": y, "class": "Se", "value": se_heatmap[y, x]})

    # 检查是否有格点未检测到原子，动态降低阈值进行补充
    def recursive_find(grid_x, grid_y, current_thresh):
        """递归降低阈值查找峰值。"""
        # 定义最低阈值限制
        min_thresh_limit = min_thresh * 0.1
        
        if current_thresh < min_thresh_limit:
            return  # 达到最低阈值限制，停止递归

        # 检查当前格点是否已有原子
        in_grid = [p for p in combined_peaks if grid_x <= p["x"] < grid_x + grid_size and grid_y <= p["y"] < grid_y + grid_size]
        if not in_grid:
            # 动态降低阈值寻找峰值
            local_te_coords = detect_peaks(te_heatmap[grid_y:grid_y + grid_size, grid_x:grid_x + grid_size], current_thresh)
            local_se_coords = detect_peaks(se_heatmap[grid_y:grid_y + grid_size, grid_x:grid_x + grid_size], current_thresh)
            if local_te_coords or local_se_coords:
                # 选择热力最高的峰
                local_peaks = []
                for lx, ly in local_te_coords:
                    local_peaks.append({"x": grid_x + lx, "y": grid_y + ly, "class": "Te", "value": te_heatmap[grid_y + ly, grid_x + lx]})
                for lx, ly in local_se_coords:
                    local_peaks.append({"x": grid_x + lx, "y": grid_y + ly, "class": "Se", "value": se_heatmap[grid_y + ly, grid_x + lx] * se_weight})
                best_peak = max(local_peaks, key=lambda p: p["value"])
                combined_peaks.append(best_peak)
            else:
                # 递归降低阈值
                recursive_find(grid_x, grid_y, current_thresh / 2)
    
    # 遍历所有格点，对没有找到原子的格点进行递归查找
    h, w = te_heatmap.shape
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            # 检查当前格点是否已有原子
            in_grid = [p for p in combined_peaks if x <= p["x"] < x + grid_size and y <= p["y"] < y + grid_size]
            if not in_grid:
                # 对没有找到原子的格点进行递归查找，从初始阈值的80%开始
                recursive_find(x, y, min_thresh * 0.8)
    
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

def calculate_accuracy(pred_coords, true_coords, threshold=5):
    """
    计算预测坐标与真实坐标的匹配准确率。
    Args:
        pred_coords (list): 预测的坐标列表 [(x1, y1), (x2, y2), ...]。
        true_coords (list): 真实的坐标列表 [(x1, y1), (x2, y2), ...]。
        threshold (float): 匹配的距离阈值。
    Returns:
        float: 准确率。
    """
    matched = 0
    for tx, ty in true_coords:
        for px, py in pred_coords:
            if np.sqrt((tx - px)**2 + (ty - py)**2) <= threshold:
                matched += 1
                break
    return matched / max(len(true_coords), 1)  # 防止除以 0

def visualize_results(img, denoised_img, img_name, pred_heatmap, pred_coords, true_coords=None):
    """
    可视化原始图像、降噪后图像、预测热力图、检测到的原子坐标以及真实的原子坐标。
    Args:
        img (numpy.ndarray): 原始灰度图像。
        denoised_img (numpy.ndarray): 降噪后的图像。
        img_name (str): 图像名称。
        pred_heatmap (numpy.ndarray): 预测的热力图 (2, H, W)。
        pred_coords (list): 预测的原子坐标 [{"x": x, "y": y, "class": "Te"}, ...]。
        true_coords (list, optional): 真实的原子坐标 [{"x": x, "y": y, "class": "Te"}, ...]。
    """
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # 显示原始图像
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 显示降噪后的图像
    axes[1].imshow(denoised_img, cmap="gray")
    axes[1].set_title("Denoised Image")
    axes[1].axis("off")

    # 显示双通道热力图（Te和Se）
    # 创建RGB热力图，Te为红色通道，Se为蓝色通道
    combined_heatmap = np.zeros((pred_heatmap.shape[1], pred_heatmap.shape[2], 3))
    combined_heatmap[:, :, 0] = pred_heatmap[0]  # Te - 红色通道
    combined_heatmap[:, :, 2] = pred_heatmap[1]  # Se - 蓝色通道
    
    axes[2].imshow(combined_heatmap)
    axes[2].set_title("Te/Se Heatmap")
    axes[2].axis("off")

    # 显示预测的原子坐标
    axes[3].imshow(denoised_img, cmap="gray")
    for atom in pred_coords:
        color = "red" if atom["class"] == "Te" else "blue"
        axes[3].scatter(atom["x"], atom["y"], c=color, s=10, label=atom["class"])
    axes[3].set_title("Predicted Atoms")
    axes[3].axis("off")

    # 显示真实的原子坐标
    axes[4].imshow(denoised_img, cmap="gray")
    if true_coords:
        for atom in true_coords:
            color = "green" if atom["class"] == "Te" else "cyan"
            axes[4].scatter(atom["x"], atom["y"], c=color, s=10, marker="x", label=f"True {atom['class']}")
    axes[4].set_title("True Atoms")
    axes[4].axis("off")

    plt.tight_layout()
    # 保存图像
    os.makedirs("test_results", exist_ok=True)
    plt.savefig(f"test_results/{img_name.replace('.png', '.jpg')}")
    plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load("checkpoints/unet_final.pth", map_location=device))
    model.eval()
    
    test_img_dir = "../dataset/test/images"
    test_label_dir = "../dataset/test/labels"
    # test_img_dir = "../raw/target/img"
    # test_label_dir = "../raw/target/labels"
    test_images = sorted(os.listdir(test_img_dir))
    
    # 初始化降噪器
    denoiser = STMDenoiser(
        filter_type='median',  # 使用中值滤波去除线噪声
        filter_size=5,         # 增大滤波核以更好地去除线噪声
        plane_subtraction=True,
        plane_order=1          # 使用平面拟合
    )
    
    total_te_accuracy = 0
    total_se_accuracy = 0
    total_te_ratio = 0
    total_se_ratio = 0
    total_pred_ratio = 0  # 用于存储预测出的两种原子数目之比的累加值
    num_images = len(test_images)
    
    for img_name in test_images:
        img_path = os.path.join(test_img_dir, img_name)
        label_path = os.path.join(test_label_dir, img_name.replace('.png', '.npy'))
        
        # 加载图像和真实标签
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        true_heatmap = np.load(label_path)  # shape: (2, H, W)
        true_te_coords = find_peaks(true_heatmap[0])
        true_se_coords = find_peaks(true_heatmap[1])
        
        # 对图像进行降噪处理
        denoised_img = denoiser.denoise_image(img, is_large_image=True)
        
        # 模型预测 - 使用降噪后的图像
        input_tensor = torch.from_numpy(denoised_img).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred_heatmap = model(input_tensor)
            pred_heatmap = torch.sigmoid(pred_heatmap).squeeze(0).cpu().numpy()
        
        pred_heatmap[0] = (pred_heatmap[0] - pred_heatmap[0].min()) / (pred_heatmap[0].max() - pred_heatmap[0].min())
        pred_heatmap[1] = (pred_heatmap[1] - pred_heatmap[1].min()) / (pred_heatmap[1].max() - pred_heatmap[1].min())
        
        # 使用联合寻峰算法
        pred_coords = find_peaks_joint(pred_heatmap[0], pred_heatmap[1])
        
        # 分离 Te 和 Se 原子
        pred_te_coords = [(atom["x"], atom["y"]) for atom in pred_coords if atom["class"] == "Te"]
        pred_se_coords = [(atom["x"], atom["y"]) for atom in pred_coords if atom["class"] == "Se"]
        
        # 计算准确率
        te_accuracy = calculate_accuracy(pred_te_coords, true_te_coords)
        se_accuracy = calculate_accuracy(pred_se_coords, true_se_coords)
        total_te_accuracy += te_accuracy
        total_se_accuracy += se_accuracy

        # 计算原子比例
        total_te_ratio += len(pred_te_coords) / max(len(true_te_coords), 1)
        total_se_ratio += len(pred_se_coords) / max(len(true_se_coords), 1)

        # 计算预测出的两种原子数目之比
        if len(pred_se_coords) > 0:  # 防止除以 0
            pred_ratio = len(pred_te_coords) / len(pred_se_coords)
        else:
            pred_ratio = float('inf')  # 如果 Se 原子数为 0，设置为无穷大
        total_pred_ratio += pred_ratio

        # 打印测试结果
        print(f"Image: {img_name}")
        print(f"True Te atoms: {len(true_te_coords)}, Pred Te atoms: {len(pred_te_coords)}")
        print(f"True Se atoms: {len(true_se_coords)}, Pred Se atoms: {len(pred_se_coords)}")
        print(f"Te accuracy: {te_accuracy:.2f}, Se accuracy: {se_accuracy:.2f}")
        print(f"Te ratio: {len(pred_te_coords) / max(len(true_te_coords), 1):.2f}, Se ratio: {len(pred_se_coords) / max(len(true_se_coords), 1):.2f}")
        print(f"Predicted Te/Se ratio: {pred_ratio:.2f}")
        print("-" * 50)
        
        # 将 true_te_coords 和 true_se_coords 转换为包含 "class" 键的字典列表
        true_coords = [{"x": x, "y": y, "class": "Te"} for x, y in true_te_coords] + \
                      [{"x": x, "y": y, "class": "Se"} for x, y in true_se_coords]

        # 可视化结果
        visualize_results(img, denoised_img, img_name, pred_heatmap, pred_coords, true_coords=true_coords)
    
    # 打印总体准确率和原子比例均值
    print(f"Average Te accuracy: {total_te_accuracy / num_images:.2f}")
    print(f"Average Se accuracy: {total_se_accuracy / num_images:.2f}")
    print(f"Average Te ratio: {total_te_ratio / num_images:.2f}")
    print(f"Average Se ratio: {total_se_ratio / num_images:.2f}")
    print(f"Average Predicted Te/Se ratio: {total_pred_ratio / num_images:.2f}")

if __name__ == "__main__":
    main()