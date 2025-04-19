import json
import os
import re
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
    # Path to the directory containing JSON files
    dir_path = r"c:\Users\Wangq1h\Desktop\bysj\AI+STM\Counting\pytorch-project\raw\processed"
    
    # Ensure the directory exists
    if not os.path.exists(dir_path):
        print(f"Directory not found: {dir_path}")
        return
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    
    # Sort files by their numerical index
    json_files.sort(key=extract_file_index)
    
    # Initialize lists to store results
    internal_te_counts = []
    internal_doping_ratios = []
    file_indices = []
    
    # Process each file
    for file_name in json_files:
        file_path = os.path.join(dir_path, file_name)
        results = process_file(file_path, boundary_threshold=0)
        
        # Extract internal statistics
        internal_te_counts.append(results['internal']['te_count'])
        internal_doping_ratios.append(results['internal']['doping_ratio'])
        
        # Extract file index
        file_index = extract_file_index(file_name)
        file_indices.append(file_index)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Plot internal Te counts
    plt.subplot(2, 1, 1)
    plt.plot(file_indices, internal_te_counts, marker='o', label='Internal Te Count')
    plt.xlabel('File Index')
    plt.ylabel('Te Count')
    plt.title('Internal Te Count vs File Index')
    plt.grid(True)
    plt.legend()
    
    # Plot internal doping ratios
    plt.subplot(2, 1, 2)
    plt.plot(file_indices, internal_doping_ratios, marker='o', label='Internal Te Doping Ratio')
    plt.xlabel('File Index')
    plt.ylabel('Doping Ratio')
    plt.title('Internal Te Doping Ratio vs File Index')
    plt.grid(True)
    plt.legend()
    
    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()