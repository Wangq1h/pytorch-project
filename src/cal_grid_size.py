import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_lattice_constant(image_path):
    """
    对输入图像进行 FFT，绘制水平截面强度-频率曲线。
    
    Args:
        image_path (str): 输入图像的路径。
    """
    # 读取图像并转换为灰度图
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # 对图像进行快速傅里叶变换 (FFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)  # 将低频移动到中心
    magnitude_spectrum = np.abs(fshift)  # 计算频谱的幅值

    # 找到频谱图的中心
    center = (magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2)
    print(f"Center of the FFT magnitude spectrum: {center}")

    # 提取水平经过中心的截面
    horizontal_section = magnitude_spectrum[center[0], :]  # 水平截面

    # 绘制水平截面强度-频率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(horizontal_section, label="Horizontal Section")
    plt.xlabel("Frequency (pixels)")
    plt.ylabel("Intensity")
    plt.title("Horizontal Section of FFT Magnitude Spectrum")
    plt.xlim(200,300)
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 输入图像路径
    image_path = '../raw/target/images/2-Vortex6_3.png'

    # 绘制水平截面强度-频率曲线
    try:
        calculate_lattice_constant(image_path)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()