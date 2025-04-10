import os
import json
import cv2
import numpy as np
import torch
from models.unet import UNet
from test import find_peaks_joint  # 引入联合寻峰算法

def infer_and_generate_json(image_path, model_path, output_json_path, device="cuda"):
    """
    使用训练好的模型对输入图像进行推理，并生成原子坐标 JSON 文件。
    Args:
        image_path (str): 输入图像路径。
        model_path (str): 训练好的模型权重路径。
        output_json_path (str): 输出 JSON 文件路径。
        device (str): 使用的设备 ("cuda" 或 "cpu")。
    """
    # 加载模型
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        pred_heatmap = model(input_tensor)
        pred_heatmap = torch.sigmoid(pred_heatmap).squeeze(0).cpu().numpy()

    # 归一化热力图
    pred_heatmap[0] = (pred_heatmap[0] - pred_heatmap[0].min()) / (pred_heatmap[0].max() - pred_heatmap[0].min())
    pred_heatmap[1] = (pred_heatmap[1] - pred_heatmap[1].min()) / (pred_heatmap[1].max() - pred_heatmap[1].min())

    # 使用联合寻峰算法
    atom_coords = find_peaks_joint(
        te_heatmap=pred_heatmap[0],
        se_heatmap=pred_heatmap[1],
        grid_size=15,  # 原子阵列的格点大小
        min_thresh=0.1,  # 最小阈值
        nms_ksize=5,  # 非极大值抑制核大小
        min_distance=10,  # 最小峰值间距
    )

    # 生成 JSON 数据
    atom_data = [{"x": int(atom["x"]), "y": int(atom["y"]), "class": atom["class"]} for atom in atom_coords]

    # 保存 JSON 文件
    with open(output_json_path, "w") as f:
        json.dump(atom_data, f, indent=4)
    print(f"Generated JSON file: {output_json_path}")

# 示例调用
if __name__ == "__main__":
    image_path = "../raw/target"
    model_path = "./checkpoints/unet_final.pth"
    output_json_path = "../raw/target"
    os.makedirs(output_json_path, exist_ok=True)

    for image_file in os.listdir(image_path):
        if image_file.endswith(('.png', '.jpg', '.jpeg')):
            input_image_path = os.path.join(image_path, image_file)
            output_json_file = os.path.splitext(image_file)[0] + ".json"
            output_json_path_full = os.path.join(output_json_path, output_json_file)
            infer_and_generate_json(
                image_path=input_image_path,
                model_path=model_path,
                output_json_path=output_json_path_full
            )