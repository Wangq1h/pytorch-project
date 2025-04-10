import os
import numpy as np
import cv2
from scipy import stats
from scipy.optimize import least_squares
from scipy.ndimage import median_filter

class STMDenoiser:
    def __init__(self, filter_type='median', filter_size=3, filter_sigma=1.0, 
                 plane_subtraction=True, plane_order=1):
        """
        初始化STM图像降噪器
        filter_type: 滤波类型，可选 'gaussian', 'median', 'bilateral'
        filter_size: 滤波核大小
        filter_sigma: 高斯滤波标准差或双边滤波的颜色空间标准差
        plane_subtraction: 是否进行平面减去
        plane_order: 平面拟合的阶数，1为平面，2为二次曲面，以此类推
        """
        self.filter_type = filter_type
        self.filter_size = filter_size
        self.filter_sigma = filter_sigma
        self.plane_subtraction = plane_subtraction
        self.plane_order = plane_order

    def fit_surface(self, image):
        """
        使用多项式拟合图像表面
        image: 2D numpy数组
        returns: 拟合的参数
        """
        h, w = image.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # 将坐标和值展平
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = image.flatten()
        
        # 构建多项式特征
        if self.plane_order == 1:
            # 平面拟合: z = ax + by + c
            A = np.vstack([X_flat, Y_flat, np.ones_like(X_flat)]).T
        elif self.plane_order == 2:
            # 二次曲面拟合: z = ax² + by² + cxy + dx + ey + f
            A = np.vstack([X_flat**2, Y_flat**2, X_flat*Y_flat, 
                           X_flat, Y_flat, np.ones_like(X_flat)]).T
        else:
            # 默认使用平面拟合
            A = np.vstack([X_flat, Y_flat, np.ones_like(X_flat)]).T
        
        # 使用最小二乘法拟合
        coeffs = np.linalg.lstsq(A, Z_flat, rcond=None)[0]
        return coeffs

    def subtract_surface(self, image):
        """
        减去拟合的表面
        image: 2D numpy数组
        returns: 减去表面后的图像
        """
        if not self.plane_subtraction:
            return image
            
        h, w = image.shape
        x = np.arange(w)
        y = np.arange(h)
        X, Y = np.meshgrid(x, y)
        
        # 拟合表面
        coeffs = self.fit_surface(image)
        
        # 计算表面
        if self.plane_order == 1:
            # 平面: z = ax + by + c
            a, b, c = coeffs
            surface = a * X + b * Y + c
        elif self.plane_order == 2:
            # 二次曲面: z = ax² + by² + cxy + dx + ey + f
            a, b, c, d, e, f = coeffs
            surface = a * X**2 + b * Y**2 + c * X * Y + d * X + e * Y + f
        else:
            # 默认使用平面
            a, b, c = coeffs
            surface = a * X + b * Y + c
        
        # 减去表面
        result = image - surface
        
        # 归一化结果到[0,1]范围
        min_val = np.min(result)
        max_val = np.max(result)
        if max_val > min_val:
            result = (result - min_val) / (max_val - min_val)
        
        return result

    def apply_filter(self, image):
        """
        应用滤波
        image: 2D numpy数组
        returns: 滤波后的图像
        """
        # 确保filter_size是奇数
        if self.filter_size % 2 == 0:
            self.filter_size += 1
            
        if self.filter_type == 'gaussian':
            # 高斯滤波
            filtered = cv2.GaussianBlur(image, (self.filter_size, self.filter_size), self.filter_sigma)
        elif self.filter_type == 'median':
            # 中值滤波 - 对去除线噪声特别有效
            filtered = median_filter(image, size=self.filter_size)
        elif self.filter_type == 'bilateral':
            # 双边滤波 - 保留边缘的同时平滑区域
            filtered = cv2.bilateralFilter(image, self.filter_size, self.filter_sigma*75, self.filter_sigma*75)
        else:
            # 默认使用中值滤波
            filtered = median_filter(image, size=self.filter_size)
            
        return filtered

    def denoise_image(self, image, is_large_image=True):
        """
        对图像进行降噪处理
        image: numpy数组，灰度图像，值范围[0,1]
        is_large_image: 是否为大图像，决定是否进行平面减去
        """
        # 保存原始图像的最小值和最大值
        original_min = np.min(image)
        original_max = np.max(image)
        
        # 1. 减去表面（仅对大图像）
        if is_large_image and self.plane_subtraction:
            image_no_surface = self.subtract_surface(image)
        else:
            image_no_surface = image
        
        # 2. 应用滤波
        denoised = self.apply_filter(image_no_surface)
        
        # 3. 确保结果在原始图像的值范围内
        denoised_min = np.min(denoised)
        denoised_max = np.max(denoised)
        
        if denoised_max > denoised_min:
            # 归一化到[0,1]
            denoised = (denoised - denoised_min) / (denoised_max - denoised_min)
            # 映射回原始范围
            denoised = denoised * (original_max - original_min) + original_min
        
        return denoised

def test_denoising():
    """
    测试降噪效果
    """
    # 初始化降噪器 - 使用中值滤波，对大图像进行平面减去
    denoiser = STMDenoiser(
        filter_type='median',  # 使用中值滤波去除线噪声
        filter_size=5,         # 增大滤波核以更好地去除线噪声
        plane_subtraction=True,
        plane_order=1          # 使用平面拟合
    )
    
    # 读取测试图像
    # test_image_path = '../data/Vortex1_Z.png'
    test_image_path = '../data/images/FTS0005_0_0.png'
    img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    
    # 降噪 - 标记为大图像
    denoised_img = denoiser.denoise_image(img, is_large_image=True)
    
    # 将图像转换回uint8格式用于显示和保存
    img_display = (img * 255).astype(np.uint8)
    denoised_display = (denoised_img * 255).astype(np.uint8)
    
    # 创建并显示对比图
    h, w = img_display.shape
    comparison = np.zeros((h, w*2), dtype=np.uint8)
    comparison[:, :w] = img_display
    comparison[:, w:] = denoised_display
    
    # 放大图像用于显示
    scale_factor = 4  # 放大倍数
    comparison_large = cv2.resize(comparison, (w*2*scale_factor, h*scale_factor), 
                                 interpolation=cv2.INTER_NEAREST)
    
    # 显示结果
    cv2.imshow('Original vs Denoised (Zoomed)', comparison_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    output_dir = '../data/denoised'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'denoised_FTS0005_0_0.png')
    cv2.imwrite(output_path, denoised_display)
    print(f"Saved denoised image to {output_path}")

if __name__ == '__main__':
    test_denoising() 