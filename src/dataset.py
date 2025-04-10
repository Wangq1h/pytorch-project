import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from denoise import STMDenoiser

class FeTeSeDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, denoise=True):
        """
        img_dir: 原始STM图像文件夹
        label_dir: 存放热力图标签的文件夹
        transform: 图像增强或预处理的transform
        denoise: 是否对图像进行降噪处理
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.denoise = denoise
        
        # 初始化降噪器
        if denoise:
            self.denoiser = STMDenoiser(
                filter_type='median',  # 使用中值滤波去除线噪声
                filter_size=5,         # 增大滤波核以更好地去除线噪声
                plane_subtraction=True,
                plane_order=1          # 使用平面拟合
            )
        
        # 假设 img_dir 下有若干 .png 文件
        self.img_names = sorted([
            f for f in os.listdir(img_dir) if f.endswith('.png')
        ])
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.npy'))
        
        # 读取灰度图, shape = (H, W)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 转为 float32, 归一化到 [0,1]
        img = img.astype(np.float32) / 255.0
        
        # 对图像进行降噪处理
        if self.denoise:
            img = self.denoiser.denoise_image(img, is_large_image=True)
            # 确保数据类型为 float32
            img = img.astype(np.float32)
        
        # 读取对应的热力图标签, shape = (2, H, W)
        heatmap = np.load(label_path)  # (2, 512, 512)
        heatmap = heatmap.astype(np.float32)
        
        if self.transform is not None:
            # 这里可根据需要进行旋转、翻转等数据增强
            img, heatmap = self.transform(img, heatmap)
        
        # 转成 PyTorch 张量，确保使用 float32
        img = torch.from_numpy(img).unsqueeze(0).float()  # 显式指定 float32
        heatmap = torch.from_numpy(heatmap).float()  # 显式指定 float32
        
        return img, heatmap