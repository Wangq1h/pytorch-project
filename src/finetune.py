import cv2
import json
import os
import shutil  # 用于移动文件
import numpy as np  # 用于处理热力图

# 全局变量
annotations = []
current_image = None
selected_atom = None  # 当前选中的原子索引
zoom_factor = 1  # 固定放大倍数（128x128 -> 512x512）
pan_offset = [0, 0]  # 平移偏移量
start_pan = None  # 用于记录鼠标拖动的起始点

def load_annotations(json_path):
    """加载 JSON 文件中的原子坐标。"""
    global annotations
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            annotations = json.load(f)
    else:
        annotations = []

def save_annotations(json_path):
    """保存原子坐标到 JSON 文件。"""
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)
    print(f"Saved annotations to {json_path}")

def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于交互操作。"""
    global annotations, selected_atom, start_pan, pan_offset

    # 将点击坐标映射回原始图像坐标
    orig_x = int((x + pan_offset[0]) / zoom_factor)
    orig_y = int((y + pan_offset[1]) / zoom_factor)

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击，新增硒原子
        annotations.append({"x": orig_x, "y": orig_y, "class": "Se"})
        selected_atom = None

    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键点击，新增碲原子
        annotations.append({"x": orig_x, "y": orig_y, "class": "Te"})
        selected_atom = None

    elif event == cv2.EVENT_MOUSEMOVE:  # 鼠标移动，选中原子
        for i, atom in enumerate(annotations):
            if abs(atom["x"] - orig_x) < 2 and abs(atom["y"] - orig_y) < 2:
                selected_atom = i
                return
        selected_atom = None

    elif event == cv2.EVENT_MBUTTONDOWN:  # 中键按下，开始平移
        start_pan = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_MBUTTON:  # 中键拖动，平移
        if start_pan is not None:
            dx = x - start_pan[0]
            dy = y - start_pan[1]
            pan_offset[0] -= dx
            pan_offset[1] -= dy
            start_pan = (x, y)

    elif event == cv2.EVENT_MBUTTONUP:  # 中键释放，停止平移
        start_pan = None

def visualize(image_path, json_path, output_dir):
    """加载图像和 JSON 文件，并启动交互式界面。"""
    global current_image, selected_atom, pan_offset

    # 加载图像
    current_image = cv2.imread(image_path)
    if current_image is None:
        print(f"Failed to load image: {image_path}")
        return

    # 将图像调整为 128x128
    current_image = cv2.resize(current_image, (512, 512))
    pan_offset = [0, 0]  # 重置平移偏移量

    # 创建窗口
    cv2.namedWindow("Interactive Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Interactive Visualization", 1024, 1024)
    cv2.setMouseCallback("Interactive Visualization", mouse_callback)

    # 加载标注
    load_annotations(json_path)

    while True:
        # 放大图像
        height, width = current_image.shape[:2]
        resized_image = cv2.resize(current_image, (width * zoom_factor, height * zoom_factor))

        # 平移图像
        canvas = resized_image[
            max(0, pan_offset[1]):min(height * zoom_factor, pan_offset[1] + 512),
            max(0, pan_offset[0]):min(width * zoom_factor, pan_offset[0] + 512)
        ]

        # 如果平移超出范围，用黑色填充
        canvas_padded = cv2.copyMakeBorder(
            canvas,
            top=max(0, 512 - canvas.shape[0]),
            bottom=0,
            left=max(0, 512 - canvas.shape[1]),
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # 在图像上绘制标注点
        for atom in annotations:
            color = (0, 255, 0) if atom["class"] == "Se" else (0, 0, 255)  # 绿色: Se, 红色: Te
            cv2.circle(canvas_padded,
                       (int(atom["x"] * zoom_factor - pan_offset[0]),
                        int(atom["y"] * zoom_factor - pan_offset[1])),
                       5, color, -1)

        # 显示图像
        cv2.imshow("Interactive Visualization", canvas_padded)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):  # 按 's' 保存标注并移动文件
            save_annotations(json_path)
            # 移动图片和 JSON 文件到目标目录
            shutil.move(image_path, os.path.join(output_dir, os.path.basename(image_path)))
            shutil.move(json_path, os.path.join(output_dir, os.path.basename(json_path)))
            print(f"Moved {image_path} and {json_path} to {output_dir}")
            break
        elif key == ord("q"):  # 按 'q' 退出
            print("Exiting...")
            break
        elif key == ord("r") and selected_atom is not None:  # 按 'r' 删除选中的原子
            annotations.pop(selected_atom)
            selected_atom = None

    cv2.destroyAllWindows()

def find_peaks_joint(te_heatmap, se_heatmap, grid_size=15, min_thresh=0.1, nms_ksize=5, min_distance=10):
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

    # 按热力值排序
    combined_peaks = sorted(combined_peaks, key=lambda p: p["value"], reverse=True)
    print(f"Detected {len(combined_peaks)} peaks before filtering.")

    # 应用最小距离约束，确保同一格点内只保留热力最高的峰
    filtered_peaks = []
    for peak in combined_peaks:
        x, y = peak["x"], peak["y"]
        if all(np.sqrt((x - fp["x"])**2 + (y - fp["y"])**2) >= min_distance for fp in filtered_peaks):
            filtered_peaks.append(peak)
    print(f"Filtered to {len(filtered_peaks)} peaks after distance filtering.")
    # 检查是否有格点未检测到原子，动态降低阈值进行补充
    for grid_x in range(0, te_heatmap.shape[1], grid_size):
        for grid_y in range(0, te_heatmap.shape[0], grid_size):
            # 检查当前格点是否已有原子
            in_grid = [p for p in filtered_peaks if grid_x <= p["x"] < grid_x + grid_size and grid_y <= p["y"] < grid_y + grid_size]
            if not in_grid:
                # 动态降低阈值寻找峰值
                local_te_coords = detect_peaks(te_heatmap[grid_y:grid_y + grid_size, grid_x:grid_x + grid_size], min_thresh / 2)
                local_se_coords = detect_peaks(se_heatmap[grid_y:grid_y + grid_size, grid_x:grid_x + grid_size], min_thresh / 2)
                if local_te_coords or local_se_coords:
                    # 选择热力最高的峰
                    local_peaks = []
                    for lx, ly in local_te_coords:
                        local_peaks.append({"x": grid_x + lx, "y": grid_y + ly, "class": "Te", "value": te_heatmap[grid_y + ly, grid_x + lx]})
                    for lx, ly in local_se_coords:
                        local_peaks.append({"x": grid_x + lx, "y": grid_y + ly, "class": "Se", "value": se_heatmap[grid_y + ly, grid_x + lx]})
                    best_peak = max(local_peaks, key=lambda p: p["value"])
                    filtered_peaks.append(best_peak)

    # 返回最终的峰值列表
    return [{"x": p["x"], "y": p["y"], "class": p["class"]} for p in filtered_peaks]

def process_heatmaps(te_heatmap, se_heatmap):
    """
    处理热力图，调用 find_peaks_joint 确定原子坐标。
    Args:
        te_heatmap (numpy.ndarray): Te 原子的热力图。
        se_heatmap (numpy.ndarray): Se 原子的热力图。
    Returns:
        list: 原子坐标及种类 [{"x": x, "y": y, "class": "Te"}, {"x": x, "y": y, "class": "Se"}, ...]。
    """
    return find_peaks_joint(te_heatmap, se_heatmap)

# 示例调用
if __name__ == "__main__":
    image_path = "../raw/target/finetune/images"
    json_path = "../raw/target/finetune/labels"
    output_dir = "../raw/processed"  # 存放修正后的图片和 JSON 文件的目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历文件夹中的所有图片
    for file_name in os.listdir(image_path):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            image_file_path = os.path.join(image_path, file_name)
            json_file_path = os.path.join(json_path, os.path.splitext(file_name)[0] + ".json")
            visualize(image_file_path, json_file_path, output_dir)
            print(f"Processed {file_name}")
