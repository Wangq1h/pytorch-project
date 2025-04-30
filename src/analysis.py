import json
import os
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

def delaunay_boundary_detection(coords, image_shape=(512,512), margin=0.5):
    """
    基于 Delaunay 三角剖分区分边界点和内部点，并考虑图像边缘的点。
    Args:
        coords (list): 原子坐标列表，每个坐标是一个字典，包含 "x" 和 "y"。
        image_shape (tuple): 图像的形状 (height, width)，用于检测图像边缘的点。
        margin (int): 边缘点的检测范围。
    Returns:
        tuple: (内部点列表, 边界点列表)
    """
    # 提取点的坐标
    points = np.array([[coord["x"], coord["y"]] for coord in coords])
    
    # 首先检测图像边缘的点
    edge_points = set()
    if image_shape is not None:
        height, width = image_shape
        for i, point in enumerate(points):
            if (
                point[0] < margin or point[0] > width - margin or
                point[1] < margin or point[1] > height - margin
            ):
                edge_points.add(i)
    
    # 进行 Delaunay 三角剖分
    tri = Delaunay(points)
    
    # 统计边的出现次数
    edge_count = {}
    for simplex in tri.simplices:
        edges = [
            tuple(sorted([simplex[0], simplex[1]])),
            tuple(sorted([simplex[1], simplex[2]])),
            tuple(sorted([simplex[2], simplex[0]]))
        ]
        for edge in edges:
            if edge in edge_count:
                edge_count[edge] += 1
            else:
                edge_count[edge] = 1
    
    # 提取边界边（出现次数为 1 的边）
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    # 提取边界点，但排除已经在边缘点集合中的点
    for edge in boundary_edges:
        for point_idx in edge:
            if point_idx not in edge_points:
                edge_points.add(point_idx)
    
    # 提取内部点
    all_points = set(range(len(points)))
    internal_points = all_points - edge_points
    
    # 将点索引转换回原始坐标
    boundary_coords = [coords[i] for i in edge_points]
    internal_coords = [coords[i] for i in internal_points]
    
    return internal_coords, boundary_coords

def process_file(file_path):
    """
    Process a single JSON file to classify points and calculate statistics.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # 使用 Delaunay 三角剖分区分边界点和内部点
    center_coords, border_coords = delaunay_boundary_detection(data)
    
    # 统计内部和边界原子的数量
    center_te_count = sum(1 for coord in center_coords if coord["class"] == "Te")
    center_se_count = sum(1 for coord in center_coords if coord["class"] == "Se")
    
    # 计算 Te 的掺杂比例
    center_total_atoms = center_te_count + center_se_count
    center_te_ratio = (center_te_count / center_total_atoms) * 100 if center_total_atoms > 0 else 0
    
    return {
        "center_te_count": center_te_count,
        "center_te_ratio": center_te_ratio
    }

def read_infer_results(results_file):
    """
    Read the results from infer.py's results.txt file.
    """
    te_counts = []
    doping_ratios = []
    file_indices = []
    
    with open(results_file, 'r') as f:
        for line in f:
            if "File:" in line:
                # Extract file index
                file_index = int(line.split("File:")[1].split("-")[0].strip())
                file_indices.append(file_index)
                
                # Extract Center Te count and ratio
                parts = line.split("Center - Te:")
                if len(parts) > 1:
                    te_count = int(parts[1].split(",")[0].strip())
                    ratio_part = line.split("Te Ratio:")[1].split("%")[0].strip()
                    ratio = float(ratio_part)
                    
                    te_counts.append(te_count)
                    doping_ratios.append(ratio)
    
    return file_indices, te_counts, doping_ratios

def main():
    # 数据源路径
    dir_path = '../raw/processed'
    infer_results_file = '../raw/target/results/results.txt'
    
    # 确保目录和文件存在
    if not os.path.exists(dir_path) or not os.path.exists(infer_results_file):
        print(f"Directory or file not found: {dir_path} or {infer_results_file}")
        return
    
    # 获取JSON文件
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    json_files.sort(key=lambda x: int(x.split('-')[0]))
    
    # 初始化列表存储结果
    analysis_te_counts = []
    analysis_doping_ratios = []
    file_indices = []
    
    # 处理analysis.py的结果
    for file_name in json_files:
        file_path = os.path.join(dir_path, file_name)
        results = process_file(file_path)
        
        analysis_te_counts.append(results['center_te_count'])
        analysis_doping_ratios.append(results['center_te_ratio'])
        
        file_index = int(file_name.split('-')[0])
        file_indices.append(file_index)
    
    # 读取infer.py的结果
    infer_indices, infer_te_counts, infer_doping_ratios = read_infer_results(infer_results_file)

    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制Te数量对比
    plt.subplot(2, 1, 1)
    plt.plot(file_indices, analysis_te_counts, 'b-o', label='human annotation')
    plt.plot(infer_indices, infer_te_counts, 'r-o', label='model prediction')
    plt.xlabel('File Index')
    plt.ylabel('Center Te Count')
    plt.title('Center Te Count Comparison')
    plt.grid(True)
    plt.legend()
    
    # 绘制掺杂比例对比
    plt.subplot(2, 1, 2)
    plt.plot(infer_indices, infer_doping_ratios, 'r-o', label='Infer.py')
    plt.plot(file_indices, analysis_doping_ratios, 'b-o', label='Analysis.py')
    plt.xlabel('File Index')
    plt.ylabel('Doping Ratio (%)')
    plt.title('Te Doping Ratio')
    plt.grid(True)
    plt.legend()
    
    # 调整布局并显示
    plt.tight_layout()
    output_dir = os.path.dirname(dir_path)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'comparison_results.png'))
    plt.show()
    
    infer_te_counts.pop(15)
    infer_doping_ratios.pop(15)
    infer_indices.pop(15)

    # 创建DataFrame保存数据
    data = {
        'File_Index': file_indices,
        'Human_Annotation_Te_Count': analysis_te_counts,
        'Human_Annotation_Te_Ratio': analysis_doping_ratios,
        'Model_Prediction_Te_Count': infer_te_counts,
        'Model_Prediction_Te_Ratio': infer_doping_ratios
    }
    df = pd.DataFrame(data)
    
    # # 保存数据到CSV文件
    
    csv_path = os.path.join(output_dir, 'plot_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Plot data has been saved to: {csv_path}")
    
    

if __name__ == "__main__":
    main()