import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FeTeSeDataset
from models.unet import UNet

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    print(f"Model is on device: {next(model.parameters()).device}")
    model.train()
    # for img, heatmap in dataloader:
    #     print(f"Image is on device: {img.device}")
    #     print(f"Heatmap is on device: {heatmap.device}")
    #     break
    total_loss = 0.0
    for img, heatmap in dataloader:
        img = img.to(device, non_blocking=True)  # 使用 non_blocking=True 加速数据传输
        heatmap = heatmap.to(device, non_blocking=True)    # shape: (B, 2, H, W)
        
        # print(f"Image is on device: {img.device}")
        # print(f"Heatmap is on device: {heatmap.device}")

        # 检查数据是否包含 NaN 或 Inf
        assert not torch.isnan(img).any(), "Input image contains NaN"
        assert not torch.isnan(heatmap).any(), "Heatmap contains NaN"
        
        optimizer.zero_grad()
        pred = model(img)               # shape: (B, 2, H, W)
        
        # 添加平滑项，避免全 0 的情况
        epsilon = 1e-7
        heatmap = heatmap.clamp(min=epsilon, max=1-epsilon)
        
        loss = criterion(pred, heatmap)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for img, heatmap in dataloader:
            img = img.to(device, non_blocking=True)
            heatmap = heatmap.to(device, non_blocking=True)
            
            pred = model(img)
            
            epsilon = 1e-7
            heatmap = heatmap.clamp(min=epsilon, max=1-epsilon)
            
            loss = criterion(pred, heatmap)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    img_dir = "/dev/shm/dataset/train/images"
    label_dir = "/dev/shm/dataset/train/labels"
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = 16
    lr = 1e-3
    num_epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建训练集和测试集
    train_dataset = FeTeSeDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=None,
        denoise=True
    )
    test_dataset = FeTeSeDataset(
        img_dir="/dev/shm/dataset/test/images",
        label_dir="/dev/shm/dataset/test/labels",
        transform=None,
        denoise=True
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    model = UNet(in_channels=1, out_channels=2).to(device)
    # print(f"Model is on device: {next(model.parameters()).device}")
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 初始化权重
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    model.apply(init_weights)
    
    # 打开文件保存 loss
    loss_file_path = os.path.join(save_dir, "training_loss.txt")
    with open(loss_file_path, "w") as loss_file:
        loss_file.write("Epoch,Train_Loss,Test_Loss\n")
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            
            # 在测试集上评估
            test_loss = evaluate(model, test_loader, criterion, device)
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            # 保存 loss 到文件
            loss_file.write(f"{epoch+1},{train_loss:.4f},{test_loss:.4f}\n")
            
            # 每 100 个 epoch 保存一次模型
            if (epoch + 1) % 100 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"unet_epoch_{epoch+1}.pth"))
        
        # 最终保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, "unet_final.pth"))

if __name__ == "__main__":
    main()