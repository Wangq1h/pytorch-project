import os
import json
import cv2
import numpy as np
import random

def augment_image_and_labels(image_path, json_path, output_dir, num_augmentations=5, scale_range=(0.5, 1.5), rotation_range=(0, 90), target_size=128):
    """
    对图像和对应的坐标 JSON 文件进行随机旋转和缩放操作，并保存结果。
    
    Args:
        image_path (str): 输入图像路径。
        json_path (str): 输入 JSON 文件路径。
        output_dir (str): 输出目录路径。
        num_augmentations (int): 每张图片生成的增强图像数量。
        scale_range (tuple): 缩放比例范围 (min_scale, max_scale)。
        rotation_range (tuple): 旋转角度范围 (min_angle, max_angle)。
        target_size (int): 输出图像的目标大小（宽和高）。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    # 读取 JSON 文件
    with open(json_path, "r") as f:
        coords = json.load(f)

    def crop_or_pad_image_and_coords(image, coords, target_size):
        """
        将图像裁剪或填补到目标大小，同时调整坐标。
        """
        h, w = image.shape
        padded_image = np.zeros((target_size, target_size), dtype=image.dtype)
        offset_x = max((target_size - w) // 2, 0)
        offset_y = max((target_size - h) // 2, 0)
        crop_x = max((w - target_size) // 2, 0)
        crop_y = max((h - target_size) // 2, 0)

        # 填补或裁剪图像
        padded_image[offset_y:offset_y + min(h, target_size), offset_x:offset_x + min(w, target_size)] = \
            image[crop_y:crop_y + min(h, target_size), crop_x:crop_x + min(w, target_size)]

        # 调整坐标
        adjusted_coords = []
        for coord in coords:
            x, y = coord["x"], coord["y"]
            new_x = x - crop_x + offset_x
            new_y = y - crop_y + offset_y
            # 过滤超出边界的点
            if 0 <= new_x < target_size and 0 <= new_y < target_size:
                adjusted_coords.append({"x": new_x, "y": new_y, "class": coord["class"]})
        return padded_image, adjusted_coords

    # 随机生成增强图像
    for i in range(num_augmentations):
        # 随机生成缩放比例和旋转角度
        scale = random.uniform(*scale_range)
        angle = random.uniform(*rotation_range)

        # 缩放图像
        new_width = int(width * scale)
        new_height = int(height * scale)
        scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # 缩放坐标
        scaled_coords = []
        for coord in coords:
            x, y = coord["x"], coord["y"]
            new_x = int(x * scale)
            new_y = int(y * scale)
            scaled_coords.append({"x": new_x, "y": new_y, "class": coord["class"]})

        # 计算旋转矩阵
        center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 旋转图像
        rotated_image = cv2.warpAffine(scaled_image, rotation_matrix, (new_width, new_height))

        # 旋转坐标
        rotated_coords = []
        for coord in scaled_coords:
            x, y = coord["x"], coord["y"]
            new_x = int(rotation_matrix[0, 0] * x + rotation_matrix[0, 1] * y + rotation_matrix[0, 2])
            new_y = int(rotation_matrix[1, 0] * x + rotation_matrix[1, 1] * y + rotation_matrix[1, 2])
            rotated_coords.append({"x": new_x, "y": new_y, "class": coord["class"]})

        # 裁剪或填补图像和坐标
        final_image, final_coords = crop_or_pad_image_and_coords(rotated_image, rotated_coords, target_size)

        # 保存增强后的图像和 JSON 文件
        augmented_image_name = f"augmented_{i}_{os.path.basename(image_path)}"
        augmented_json_name = f"augmented_{i}_{os.path.basename(json_path)}"
        image_output_dir = os.path.join(output_dir, "images")
        label_output_dir = os.path.join(output_dir, "labels")
        cv2.imwrite(os.path.join(image_output_dir, augmented_image_name), final_image)
        with open(os.path.join(label_output_dir, augmented_json_name), "w") as f:
            json.dump(final_coords, f, indent=4)

    print(f"图像和坐标增强已完成，结果保存在 {output_dir}")

def process_all_files(images_dir, labels_dir, output_dir, num_augmentations=5, scale_range=(0.5, 1.5), rotation_range=(0, 90), target_size=128):
    """
    对 images 和 labels 文件夹中的所有文件执行增强操作。
    """
    for image_file in os.listdir(images_dir):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_dir, image_file)
            json_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + ".json")
            if os.path.exists(json_path):
                augment_image_and_labels(image_path, json_path, output_dir, num_augmentations, scale_range, rotation_range, target_size)

# 示例调用
images_dir = "../data/tuned/images"  # 替换为你的图像文件夹路径
labels_dir = "../data/tuned/labels"  # 替换为你的标签文件夹路径
output_dir = "../data/tuned"  # 替换为你的输出文件夹路径
process_all_files(images_dir, labels_dir, output_dir)