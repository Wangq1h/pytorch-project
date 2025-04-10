import matplotlib.pyplot as plt
import pandas as pd

# 读取训练损失数据
file_path = './checkpoints/training_loss.txt'
data = pd.read_csv(file_path)

# 提取 Epoch 和 Loss 列
epochs = data['Epoch']
loss = data['Loss']

# 设置 IEEE 风格的可视化
plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, label='Training Loss', color='blue', linewidth=1.5)

# 设置标题和标签
plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.xlim(-10, 6000)  # 设置 x 轴范围

# 设置网格线
plt.grid(True, linestyle='--', linewidth=0.5)

# 设置图例
plt.legend(fontsize=10)

# 设置刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 保存图像为高分辨率
plt.tight_layout()
plt.savefig('training_loss_ieee_style.png', dpi=300)

# 显示图像
plt.show()