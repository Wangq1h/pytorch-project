import json
import os
import re
import numpy as np
from matplotlib import pyplot as plt

def classify_points(data, boundary_threshold=1):
    """
    Classify points as 'internal' or 'external' based on their distance to the boundaries.
    """
    internal_points = []
    external_points = []
    for point in data:
        x, y = point['x'], point['y']
        if x <= boundary_threshold or y <= boundary_threshold or \
           x >= 511 - boundary_threshold or y >= 511 - boundary_threshold:
            external_points.append(point)
        else:
            internal_points.append(point)
    return internal_points, external_points

def calculate_te_statistics(data):
    """
    Calculate the total number of 'Te' atoms and the doping ratio.
    """
    total_te = sum(1 for point in data if point['class'] == 'Te')
    total_points = len(data)
    doping_ratio = total_te / total_points if total_points > 0 else 0
    return total_te, doping_ratio

def process_file(file_path, boundary_threshold=10):
    """
    Process a single JSON file to classify points and calculate statistics.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    internal_points, external_points = classify_points(data, boundary_threshold)
    
    internal_te_count, internal_doping_ratio = calculate_te_statistics(internal_points)
    external_te_count, external_doping_ratio = calculate_te_statistics(external_points)
    
    return {
        "internal": {
            "te_count": internal_te_count,
            "doping_ratio": internal_doping_ratio
        },
        "external": {
            "te_count": external_te_count,
            "doping_ratio": external_doping_ratio
        }
    }

def extract_file_index(file_name):
    """
    Extract the numerical index from the file name.
    Assumes the file name starts with a number followed by a dash.
    """
    match = re.match(r"(\d+)-", file_name)
    return int(match.group(1)) if match else float('inf')

def main():
    # 两个数据源的路径
    dir_path1 = r"c:\Users\Wangq1h\Desktop\bysj\AI+STM\Counting\pytorch-project\raw\processed"
    dir_path2 = r"c:\Users\Wangq1h\Desktop\bysj\AI+STM\Counting\pytorch-project\raw\span_results"
    
    # 确保目录存在
    if not os.path.exists(dir_path1) or not os.path.exists(dir_path2):
        print(f"Directory not found: {dir_path1} or {dir_path2}")
        return
    
    # 获取两个目录下的JSON文件
    json_files1 = [f for f in os.listdir(dir_path1) if f.endswith('.json')]
    json_files2 = [f for f in os.listdir(dir_path2) if f.endswith('.json')]
    
    # 按序号排序
    json_files1.sort(key=extract_file_index)
    json_files2.sort(key=extract_file_index)
    
    # 初始化列表存储结果
    internal_te_counts1 = []
    internal_doping_ratios1 = []
    internal_te_counts2 = []
    internal_doping_ratios2 = []
    file_indices = []
    
    # 处理第一个目录的文件
    for file_name in json_files1:
        file_path = os.path.join(dir_path1, file_name)
        results = process_file(file_path, boundary_threshold=0)
        
        internal_te_counts1.append(results['internal']['te_count'])
        internal_doping_ratios1.append(results['internal']['doping_ratio'])
        
        file_index = extract_file_index(file_name)
        if file_index not in file_indices:
            file_indices.append(file_index)
    
    # 处理第二个目录的文件
    for file_name in json_files2:
        file_path = os.path.join(dir_path2, file_name)
        results = process_file(file_path, boundary_threshold=0)
        
        internal_te_counts2.append(results['internal']['te_count'])
        internal_doping_ratios2.append(results['internal']['doping_ratio'])
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制Te数量对比
    plt.subplot(2, 1, 1)
    plt.plot(file_indices, internal_te_counts1, 'b-o', label='Original Method')
    plt.plot(file_indices, internal_te_counts2, 'r-o', label='Enhanced Method')
    plt.xlabel('File Index')
    plt.ylabel('Te Count')
    plt.title('Internal Te Count Comparison')
    plt.grid(True)
    plt.legend()
    
    # 绘制掺杂比例对比
    plt.subplot(2, 1, 2)
    plt.plot(file_indices, internal_doping_ratios1, 'b-o', label='Original Method')
    plt.plot(file_indices, internal_doping_ratios2, 'r-o', label='Enhanced Method')
    plt.xlabel('File Index')
    plt.ylabel('Doping Ratio')
    plt.title('Internal Te Doping Ratio Comparison')
    plt.grid(True)
    plt.legend()
    
    # 调整布局并显示
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(dir_path1)
    plt.savefig(os.path.join(output_dir, 'comparison_results.png'))
    plt.show()

if __name__ == "__main__":
    main()