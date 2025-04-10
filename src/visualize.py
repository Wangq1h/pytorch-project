import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_heatmap(image_dir, heatmap_dir, output_dir, num_samples=5):
    """
    可视化原始图像和对应的热力图，并保存结果。
    
    Args:
        image_dir (str): 原始图像文件夹路径。
        heatmap_dir (str): 热力图文件夹路径 (.npy 文件)。
        output_dir (str): 可视化结果保存路径。
        num_samples (int): 可视化的样本数量。
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    heatmap_files = sorted([f for f in os.listdir(heatmap_dir) if f.endswith('.npy')])
    
    # 确保图像和热力图文件数量一致
    assert len(image_files) == len(heatmap_files), "图像文件和热力图文件数量不一致！"
    
    # 可视化指定数量的样本
    for i, (img_file, heatmap_file) in enumerate(zip(image_files, heatmap_files)):
        if i >= num_samples:
            break
        
        # 加载原始图像
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 加载热力图
        heatmap_path = os.path.join(heatmap_dir, heatmap_file)
        heatmap = np.load(heatmap_path)  # shape: (2, height, width)
        te_heatmap = heatmap[0]  # 碲原子热力图
        se_heatmap = heatmap[1]  # 硒原子热力图
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        im1 = axes[1].imshow(te_heatmap, cmap='hot')
        axes[1].set_title("Te Heatmap")
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        im2 = axes[2].imshow(se_heatmap, cmap='hot')
        axes[2].set_title("Se Heatmap")
        axes[2].axis('off')
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # 保存可视化结果
        output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}_visualization.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)
        
        print(f"Saved visualization for {img_file} -> {output_path}")

if __name__ == "__main__":
    # 原始图像文件夹路径
    image_dir = "../data/images"
    
    # 热力图文件夹路径
    heatmap_dir = "../data/labels_npy"
    
    # 可视化结果保存路径
    output_dir = "../data/visualizations"
    
    # 可视化前 5 个样本
    visualize_heatmap(image_dir, heatmap_dir, output_dir, num_samples=5)