import os
import shutil
import random

def clear_directory(directory):
    """
    清空指定目录中的所有文件和子文件夹。
    Args:
        directory (str): 需要清空的目录路径。
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除子文件夹

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.75):
    """
    将数据集按指定比例划分为训练集和测试集，并存储到指定目录。

    Args:
        image_dir (str): 原始图像文件夹路径。
        label_dir (str): 原始标签文件夹路径。
        output_dir (str): 输出数据集文件夹路径。
        train_ratio (float): 训练集比例，默认为 0.75。
    """
    # 获取所有图片和对应的热力图文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    paired_files = [(f, os.path.splitext(f)[0] + ".npy") for f in image_files]

    # 检查热力图文件是否存在
    paired_files = [(img, lbl) for img, lbl in paired_files if os.path.exists(os.path.join(label_dir, lbl))]
    if not paired_files:
        print("No valid image-label pairs found in the specified directories.")
        return

    # 打乱文件顺序
    random.shuffle(paired_files)

    # 按比例划分
    train_size = int(len(paired_files) * train_ratio)
    train_files = paired_files[:train_size]
    test_files = paired_files[train_size:]

    # 创建输出文件夹并清空
    train_image_dir = os.path.join(output_dir, "train/images")
    train_label_dir = os.path.join(output_dir, "train/labels")
    test_image_dir = os.path.join(output_dir, "test/images")
    test_label_dir = os.path.join(output_dir, "test/labels")
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    clear_directory(train_image_dir)
    clear_directory(train_label_dir)
    clear_directory(test_image_dir)
    clear_directory(test_label_dir)

    # 复制文件到训练集和测试集
    for img_file, label_file in train_files:
        shutil.copy(os.path.join(image_dir, img_file), os.path.join(train_image_dir, img_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

    for img_file, label_file in test_files:
        shutil.copy(os.path.join(image_dir, img_file), os.path.join(test_image_dir, img_file))
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(test_label_dir, label_file))

    print(f"Dataset split completed. Train: {len(train_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    # 原始数据路径
    image_dir = "../data/images"
    label_dir = "../data/labels_npy"

    # 输出数据集路径
    output_dir = "../dataset"

    # 开始划分数据集
    split_dataset(image_dir, label_dir, output_dir, train_ratio=0.75)