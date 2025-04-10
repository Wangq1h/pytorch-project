import os
import cv2
import numpy as np

def split_image(image_path, output_dir, tile_size=128):
    """
    将单张图片分割成 tile_size x tile_size 的小图片并保存到指定文件夹。

    Args:
        image_path (str): 输入图片路径。
        output_dir (str): 输出文件夹路径。
        tile_size (int): 小图片的尺寸，默认为 128。
    """
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # 获取图片尺寸
    height, width, _ = img.shape

    # 确保图片可以被 tile_size 整除
    if height % tile_size != 0 or width % tile_size != 0:
        print(f"Image {image_path} size ({height}, {width}) is not divisible by {tile_size}. Skipping...")
        return
    
    print(f"Splitting image {image_path} of size ({height}, {width}) into tiles of size ({tile_size}, {tile_size})")

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 分割图片
    for i in range(0, height, tile_size):
        for j in range(0, width, tile_size):
            tile = img[i:i+tile_size, j:j+tile_size]
            tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{i}_{j}.png"
            tile_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(tile_path, tile)

    print(f"Image {image_path} has been split and saved to {output_dir}.")

def process_images(input_dir, output_dir, tile_size=128):
    """
    批量处理文件夹中的图片，将其分割成小图片。

    Args:
        input_dir (str): 输入图片文件夹路径。
        output_dir (str): 输出小图片文件夹路径。
        tile_size (int): 小图片的尺寸，默认为 128。
    """
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {input_dir}.")
        return

    # 分割每张图片
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        split_image(image_path, output_dir, tile_size)

if __name__ == "__main__":
    # 输入图片文件夹路径
    input_dir = "../raw/target"  # 替换为你的图片文件夹路径
    # 输出小图片文件夹路径
    output_dir = "../raw/target"  # 替换为你的输出文件夹路径

    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)

    # 开始处理
    process_images(input_dir, output_dir, tile_size=128)