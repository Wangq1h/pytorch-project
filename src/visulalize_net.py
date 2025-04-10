from torchviz import make_dot
from models.unet import UNet  # 确保路径正确
import torch

# 初始化模型
model = UNet(in_channels=1, out_channels=2)

# 构造输入张量
x = torch.randn(1, 1, 128, 128)

# ------------------ 可视化编码器 ------------------
# 编码器部分：从输入到 bottom 层
x1 = model.down1(x)
x2 = model.down2(model.pool(x1))
x3 = model.down3(model.pool(x2))
x4 = model.down4(model.pool(x3))
x5 = model.bottom(model.pool(x4))  # 编码器的最后输出

dot_encoder = make_dot(x5, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
# 设置布局和节点属性
dot_encoder.graph_attr.update({
    'rankdir': 'LR',   # 从左到右布局
    'nodesep': '0.1',
    'ranksep': '0.1',
})
dot_encoder.node_attr.update({
    'fontsize': '10',
    'shape': 'record',
    'width': '0.2',
    'height': '0.2',
    'margin': '0.05,0.05'
})

# 输出编码器图像
dot_encoder.format = 'png'
dot_encoder.render('unet_encoder')

# ------------------ 可视化解码器 ------------------
# 解码器部分：从 bottom 层到输出
# 注意：解码器中包含 skip connections，为了完整显示，这里仍使用对应的 skip 输出，但只生成解码器部分的图
x6 = model.up1(x5)
x6 = torch.cat([x4, x6], dim=1)
x6 = model.conv1(x6)

x7 = model.up2(x6)
x7 = torch.cat([x3, x7], dim=1)
x7 = model.conv2(x7)

x8 = model.up3(x7)
x8 = torch.cat([x2, x8], dim=1)
x8 = model.conv3(x8)

x9 = model.up4(x8)
x9 = torch.cat([x1, x9], dim=1)
x9 = model.conv4(x9)

decoder_output = model.out_conv(x9)

dot_decoder = make_dot(decoder_output, params=dict(model.named_parameters()), show_attrs=False, show_saved=False)
dot_decoder.graph_attr.update({
    'rankdir': 'LR',
    'nodesep': '0.1',
    'ranksep': '0.1',
})
dot_decoder.node_attr.update({
    'fontsize': '10',
    'shape': 'record',
    'width': '0.2',
    'height': '0.2',
    'margin': '0.05,0.05'
})

dot_decoder.format = 'png'
dot_decoder.render('unet_decoder')
