import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter
import os

def find_peaks_2d(image, min_distance=5, threshold=None):
    """
    在2D图像中查找局部最大值
    Args:
        image: 输入图像
        min_distance: 峰值之间的最小距离
        threshold: 峰值的最小值阈值
    Returns:
        峰值坐标列表 [(y, x), ...]
    """
    # 如果未指定阈值，使用图像最大值的10%
    if threshold is None:
        threshold = np.max(image) * 0.1
    
    # 使用高斯滤波平滑图像
    smoothed = gaussian_filter(image, sigma=1)
    
    # 初始化峰值列表
    peaks = []
    
    # 遍历图像（排除边缘）
    for i in range(min_distance, image.shape[0]-min_distance):
        for j in range(min_distance, image.shape[1]-min_distance):
            # 检查当前点是否大于阈值
            if smoothed[i, j] < threshold:
                continue
                
            # 获取邻域
            neighborhood = smoothed[i-min_distance:i+min_distance+1, 
                                 j-min_distance:j+min_distance+1]
            
            # 检查是否为局部最大值
            if smoothed[i, j] == np.max(neighborhood):
                peaks.append((i, j))
    
    return np.array(peaks)

class FujitaAtomDetector:
    def __init__(self, image_path, lattice_constant=4.14):
        """
        初始化Fujita原子检测器
        Args:
            image_path: STM图像路径
            lattice_constant: 晶格常数（单位：埃）
        """
        # 读取图像
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        self.height, self.width = self.original_image.shape
        
        print(f"图像尺寸: {self.width}x{self.height} 像素")
        
        # 晶格参数
        self.a0 = lattice_constant  # 晶格常数（埃）
        self.pixel_size = 0.29  # 每个像素代表的实际尺寸（埃）
        
        print(f"晶格常数: {self.a0} 埃")
        print(f"像素尺寸: {self.pixel_size} 埃/像素")
        
        # 计算波矢
        self.Qx = np.array([2*np.pi/self.a0, 0])  # x方向布拉格波矢
        self.Qy = np.array([0, 2*np.pi/self.a0])  # y方向布拉格波矢
        
        print(f"布拉格波矢 Qx: {self.Qx} 埃^-1")
        print(f"布拉格波矢 Qy: {self.Qy} 埃^-1")
        
        # 初始化位移场
        self.displacement_field = None
        self.corrected_positions = None

    def compute_fourier_transform(self):
        """计算图像的二维傅里叶变换"""
        # 进行傅里叶变换
        fft_image = fft2(self.original_image)
        fft_image_shifted = fftshift(fft_image)
        
        # 计算功率谱
        power_spectrum = np.abs(fft_image_shifted)**2
        
        # 打印傅里叶变换信息
        print(f"傅里叶变换尺寸: {fft_image.shape}")
        print(f"功率谱最大值: {np.max(power_spectrum):.2e}")
        print(f"功率谱最小值: {np.min(power_spectrum):.2e}")
        
        return fft_image_shifted, power_spectrum

    def find_bragg_peaks(self, power_spectrum):
        """在傅里叶空间中定位布拉格峰"""
        # 计算基于晶格常数的最小距离
        # 实空间晶格常数 a₀ = 4.14 埃
        # 傅里叶空间中的波矢 Q = 2π/a₀
        # 像素尺寸 pixel_size = 0.29 埃/像素
        # 因此，傅里叶空间中布拉格峰之间的最小距离（像素）为：
        min_distance = int(self.width * self.pixel_size / self.a0)
        print(f"基于晶格常数计算的布拉格峰最小距离: {min_distance} 像素")
        
        # 使用2D峰值检测，设置基于物理的最小距离
        peaks = find_peaks_2d(power_spectrum, min_distance=min_distance, threshold=np.max(power_spectrum)*0.1)
        
        print(f"检测到的布拉格峰数量: {len(peaks)}")
        
        # 计算每个峰的位置对应的波矢
        peak_positions = []
        for peak in peaks:
            # 将像素坐标转换为波矢
            kx = (peak[1] - self.width/2) * (2*np.pi/(self.width*self.pixel_size))
            ky = (peak[0] - self.height/2) * (2*np.pi/(self.height*self.pixel_size))
            peak_positions.append((kx, ky))
            
            print(f"布拉格峰位置 (kx, ky): ({kx:.2f}, {ky:.2f}) 埃^-1")
            print(f"布拉格峰强度: {power_spectrum[peak[0], peak[1]]:.2e}")
        
        return peak_positions

    def compute_displacement_field(self, fft_image):
        """计算位移场"""
        # 创建波矢网格
        kx = np.fft.fftfreq(self.width, self.pixel_size) * 2*np.pi
        ky = np.fft.fftfreq(self.height, self.pixel_size) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky)
        
        print(f"波矢网格范围: kx [{np.min(kx):.2f}, {np.max(kx):.2f}] 埃^-1")
        print(f"波矢网格范围: ky [{np.min(ky):.2f}, {np.max(ky):.2f}] 埃^-1")
        
        # 计算位移场
        ux = np.zeros_like(self.original_image, dtype=complex)
        uy = np.zeros_like(self.original_image, dtype=complex)
        
        # 对每个布拉格峰计算位移场分量
        for Q in [self.Qx, self.Qy]:
            # 计算相位
            phase = np.angle(fft_image)
            
            # 相位解缠
            unwrapped_phase = self.unwrap_phase(phase)
            
            # 计算位移场分量
            if np.allclose(Q, self.Qx):
                ux = unwrapped_phase / np.linalg.norm(Q)
            else:
                uy = unwrapped_phase / np.linalg.norm(Q)
        
        print(f"位移场范围: ux [{np.min(np.real(ux)):.2f}, {np.max(np.real(ux)):.2f}] 埃")
        print(f"位移场范围: uy [{np.min(np.real(uy)):.2f}, {np.max(np.real(uy)):.2f}] 埃")
        
        return np.real(ux), np.real(uy)

    def unwrap_phase(self, phase):
        """改进的相位解缠算法"""
        # 创建掩码，只处理有效区域
        mask = np.abs(phase) > 0.1
        
        # 初始化解缠后的相位
        unwrapped = np.zeros_like(phase)
        
        # 计算相位差分
        diff_x = np.diff(phase, axis=1)
        diff_y = np.diff(phase, axis=0)
        
        # 检测2π跳变
        jump_x = np.round(diff_x / (2*np.pi))
        jump_y = np.round(diff_y / (2*np.pi))
        
        # 累积相位跳变，只在有效区域内
        unwrapped[:, 1:] = np.cumsum(jump_x * mask[:, 1:], axis=1) * 2*np.pi
        unwrapped[1:, :] += np.cumsum(jump_y * mask[1:, :], axis=0) * 2*np.pi
        
        # 限制相位范围
        unwrapped = np.mod(unwrapped + np.pi, 2*np.pi) - np.pi
        
        print(f"相位解缠范围: [{np.min(phase):.2f}, {np.max(phase):.2f}]")
        print(f"解缠后相位范围: [{np.min(unwrapped):.2f}, {np.max(unwrapped):.2f}]")
        
        return unwrapped

    def correct_atom_positions(self):
        """校正原子位置"""
        # 计算位移场
        fft_image, _ = self.compute_fourier_transform()
        ux, uy = self.compute_displacement_field(fft_image)
        
        # 创建坐标网格
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        
        # 应用位移场校正
        corrected_x = X - ux
        corrected_y = Y - uy
        
        print(f"校正后x坐标范围: [{np.min(corrected_x):.2f}, {np.max(corrected_x):.2f}] 像素")
        print(f"校正后y坐标范围: [{np.min(corrected_y):.2f}, {np.max(corrected_y):.2f}] 像素")
        
        return corrected_x, corrected_y

    def detect_atoms(self):
        """检测原子位置"""
        # 校正原子位置
        corrected_x, corrected_y = self.correct_atom_positions()
        
        # 对图像进行高斯滤波以去除噪声
        smoothed_image = gaussian_filter(self.original_image, sigma=1)
        
        # 计算图像的统计特性
        mean_brightness = np.mean(smoothed_image)
        std_brightness = np.std(smoothed_image)
        
        print(f"图像平均亮度: {mean_brightness:.3f}")
        print(f"图像亮度标准差: {std_brightness:.3f}")
        
        # 设置Te和Se的检测阈值
        # Te原子通常比平均值亮，Se原子通常比平均值暗
        te_threshold = mean_brightness + 0.5 * std_brightness
        se_threshold = mean_brightness - 0.5 * std_brightness
        
        print(f"Te原子检测阈值: {te_threshold:.3f}")
        print(f"Se原子检测阈值: {se_threshold:.3f}")
        
        # 检测Te原子（亮点）
        te_peaks = find_peaks_2d(smoothed_image, min_distance=5, threshold=te_threshold)
        print(f"检测到的Te原子数量: {len(te_peaks)}")
        
        # 检测Se原子（暗点）
        # 反转图像以检测暗点
        inverted_image = 1.0 - smoothed_image
        se_peaks = find_peaks_2d(inverted_image, min_distance=5, threshold=1.0-se_threshold)
        print(f"检测到的Se原子数量: {len(se_peaks)}")
        
        # 将检测到的位置映射回校正后的坐标
        atom_positions = []
        atom_types = []
        
        # 处理Te原子
        for y, x in te_peaks:
            atom_positions.append((corrected_x[y, x], corrected_y[y, x]))
            atom_types.append('Te')
        
        # 处理Se原子
        for y, x in se_peaks:
            atom_positions.append((corrected_x[y, x], corrected_y[y, x]))
            atom_types.append('Se')
        
        return np.array(atom_positions), np.array(atom_types)

    def visualize_results(self, output_dir):
        """可视化结果"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 绘制原始图像
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.original_image, cmap='gray')
        plt.title('Original STM Image')
        plt.colorbar()
        
        # 绘制傅里叶变换
        _, power_spectrum = self.compute_fourier_transform()
        plt.subplot(1, 3, 2)
        plt.imshow(np.log(power_spectrum), cmap='jet')
        plt.title('Fourier Transform')
        plt.colorbar()
        
        # 绘制检测到的原子位置
        atom_positions, atom_types = self.detect_atoms()
        plt.subplot(1, 3, 3)
        plt.imshow(self.original_image, cmap='gray')
        
        # 分别绘制Te和Se原子
        te_mask = atom_types == 'Te'
        se_mask = atom_types == 'Se'
        
        plt.scatter(atom_positions[te_mask, 0], atom_positions[te_mask, 1], 
                   c='red', s=10, alpha=0.5, label='Te')
        plt.scatter(atom_positions[se_mask, 0], atom_positions[se_mask, 1], 
                   c='blue', s=10, alpha=0.5, label='Se')
        
        plt.title('Detected Atom Positions')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'atom_detection_results.png'))
        plt.close()
        
        # 保存原子位置和类型
        results = {
            'positions': atom_positions,
            'types': atom_types
        }
        np.save(os.path.join(output_dir, 'atom_positions.npy'), results)

def main():
    # 设置参数
    image_path = '../raw/target/images/1-Vortex1_2.png'  # 输入图像路径
    output_dir = '../raw/classical_results'  # 输出目录
    
    # 创建检测器实例
    detector = FujitaAtomDetector(image_path)
    
    # 检测原子位置
    atom_positions, atom_types = detector.detect_atoms()
    
    # 可视化结果
    detector.visualize_results(output_dir)
    
    print(f"检测到 {len(atom_positions)} 个原子位置")
    print(f"结果已保存到 {output_dir}")

if __name__ == "__main__":
    main() 