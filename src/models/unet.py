import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    """注意力模块，用于增强特征表示"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(2, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 空间注意力
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv1(pool)
        attention = self.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

class ResBlock(nn.Module):
    """残差块，用于增强特征传递"""
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，添加1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class EnhancedDoubleConv(nn.Module):
    """增强版的双卷积块，包含残差连接和注意力机制"""
    def __init__(self, in_channels, out_channels):
        super(EnhancedDoubleConv, self).__init__()
        self.res1 = ResBlock(in_channels, out_channels)
        self.res2 = ResBlock(out_channels, out_channels)
        self.attention = AttentionBlock(out_channels)
        
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.attention(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, use_enhanced=True):
        super(UNet, self).__init__()
        self.use_enhanced = use_enhanced
        
        # 选择使用增强版还是原始版的双卷积块
        if use_enhanced:
            DoubleConv = EnhancedDoubleConv
        else:
            DoubleConv = self._original_double_conv
        
        # Downsampling layers
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottom layer
        self.bottom = DoubleConv(512, 1024)
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def _original_double_conv(self, in_channels, out_channels):
        """原始的双卷积块，用于兼容性"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.pool(x1)
        x2 = self.down2(x2)
        x3 = self.pool(x2)
        x3 = self.down3(x3)
        x4 = self.pool(x3)
        x4 = self.down4(x4)
        
        # Bottom
        x5 = self.pool(x4)
        x5 = self.bottom(x5)
        
        # Decoder
        x6 = self.up1(x5)
        x6 = torch.cat([x4, x6], dim=1)
        x6 = self.conv1(x6)
        
        x7 = self.up2(x6)
        x7 = torch.cat([x3, x7], dim=1)
        x7 = self.conv2(x7)
        
        x8 = self.up3(x7)
        x8 = torch.cat([x2, x8], dim=1)
        x8 = self.conv3(x8)
        
        x9 = self.up4(x8)
        x9 = torch.cat([x1, x9], dim=1)
        x9 = self.conv4(x9)
        
        out = self.out_conv(x9)
        return out