import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FeTeSeDataset
from models.unet import UNet

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个 epoch。
    """
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def main():
    # 数据集路径
    img_dir = "../data/tuned/images"  # 分割后的图像路径
    label_dir = "../data/tuned/labels_npy"  # 分割后的标签路径
    save_dir = "checkpoints_tuned"  # 保存微调模型的路径
    os.makedirs(save_dir, exist_ok=True)

    # 创建loss记录文件
    loss_file_path = os.path.join(save_dir, "training_loss.txt")
    loss_file = open(loss_file_path, "w")
    loss_file.write("Epoch,Loss\n")  # 写入CSV格式的表头

    # 超参数
    batch_size = 32
    lr = 1e-4  # 微调时使用较小的学习率
    num_epochs = 50  # 微调的 epoch 数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载数据集
    tuned_dataset = FeTeSeDataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transform=None,  # 可添加数据增强
        denoise=False
    )
    tuned_loader = DataLoader(
        tuned_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    # 加载预训练模型
    model = UNet(in_channels=1, out_channels=2).to(device)
    pretrained_weights = "checkpoints/unet_final.pth"  # 替换为预训练模型的路径
    if os.path.exists(pretrained_weights):
        model.load_state_dict(torch.load(pretrained_weights, map_location=device))
        print("Loaded pretrained model weights.")
    else:
        print(f"Pretrained weights not found at {pretrained_weights}. Starting from scratch.")

    # 冻结部分参数（例如冻结编码器部分）
    for name, param in model.named_parameters():
        if "down" in name:  # 假设编码器部分的层名包含 "down"
            param.requires_grad = False
    print("Froze encoder parameters.")

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    try:
        # 开始微调
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, tuned_loader, optimizer, criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
            
            # 写入loss到文件
            loss_file.write(f"{epoch+1},{train_loss:.4f}\n")
            loss_file.flush()  # 立即写入磁盘

            # 每 10 个 epoch 保存一次模型
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"unet_tuned_epoch_{epoch+1}.pth"))

        # 最终保存模型
        torch.save(model.state_dict(), os.path.join(save_dir, "unet_tuned_final.pth"))
        print("Fine-tuning completed and model saved.")
        
    finally:
        # 确保文件被正确关闭
        loss_file.close()
        print(f"Training loss has been saved to {loss_file_path}")

if __name__ == "__main__":
    main()