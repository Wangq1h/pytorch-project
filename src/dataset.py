import os
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class FeTeSeDataset(Dataset):
    def __init__(self, img_dir, label_dir, denoise=False, augment=False):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.denoise = denoise
        self.augment = augment
        self.img_files = sorted(os.listdir(img_dir))
        self.label_files = sorted(os.listdir(label_dir))
        
        # 数据增强操作
        if self.augment:
            self.augmentations = A.Compose([
                A.RandomRotate90(),  # 随机旋转 90 度
                A.Flip(),  # 随机水平和垂直翻转
                A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.2)),  # 随机放大/缩小
                A.RandomBrightnessContrast(p=0.2),  # 随机亮度和对比度调整
            ])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])
        
        # 加载图像和标签
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0  # 归一化
        label = np.load(label_path).astype(np.float32)  # 加载热力图标签
        
        # 数据增强
        if self.augment and self.augmentations:
            augmented = self.augmentations(image=img, mask=label)
            img = augmented["image"]
            label = augmented["mask"]
        
        # 转换为张量
        img = torch.from_numpy(img).unsqueeze(0)  # 添加通道维度 (C, H, W)
        label = torch.from_numpy(label)  # 标签形状为 (2, H, W)
        
        return img, label