import numpy as np
from PIL import Image
import pywt
import matplotlib.pyplot as plt

def wavelet_compress_image(image_path, output_path, wavelet, level, threshold):
    """
    使用小波变换压缩图像
    参数：
        image_path: 输入图像路径
        output_path: 输出压缩图像路径
        wavelet: 小波类型（如 'db1', 'haar'）
        level: 小波分解层数
        threshold: 阈值，用于丢弃小系数
    """
    # 读取图像并转换为灰度
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=float)

    # 小波分解
    coeffs = pywt.wavedec2(img_array, wavelet=wavelet, level=level)

    # 计算所有系数的展平数组
    coeff_flat = []
    for c in coeffs:
        if isinstance(c, np.ndarray):
            coeff_flat.append(c.ravel())
        else:
            # c 是元组 (cH, cV, cD)
            for detail in c:
                coeff_flat.append(detail.ravel())
    coeff_array = np.concatenate(coeff_flat)

    # 设置阈值
    threshold_value = np.percentile(np.abs(coeff_array), threshold)  # 动态阈值

    # 对细节系数进行阈值处理（保留近似系数）
    compressed_coeffs = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            # 保留近似系数
            compressed_coeffs.append(coeff)
        else:
            # 对细节系数 (cH, cV, cD) 进行阈值处理
            cH, cV, cD = coeff
            cH = np.where(np.abs(cH) < threshold_value, 0, cH)
            cV = np.where(np.abs(cV) < threshold_value, 0, cV)
            cD = np.where(np.abs(cD) < threshold_value, 0, cD)
            compressed_coeffs.append((cH, cV, cD))

    # 小波重构
    reconstructed_array = pywt.waverec2(compressed_coeffs, wavelet=wavelet)
    reconstructed_array = np.clip(reconstructed_array, 0, 255)  # 限制像素值范围

    # 保存重构图像
    reconstructed_img = Image.fromarray(reconstructed_array.astype(np.uint8))
    reconstructed_img.save(output_path)

    # 计算压缩率（简化为非零系数的比例）
    total_coeffs = coeff_array.size
    non_zero_coeffs = sum(np.count_nonzero(c) if isinstance(c, np.ndarray) else sum(np.count_nonzero(d) for d in c) for c in compressed_coeffs)
    compression_ratio = (total_coeffs - non_zero_coeffs) / total_coeffs * 100

    # 显示原始和压缩图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_array, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Compressed Image\nCompression Ratio: {compression_ratio:.2f}%")
    plt.imshow(reconstructed_array, cmap='gray')
    plt.axis('off')

    plt.show()

    return compression_ratio

# 示例用法
image_path = r"C:\Users\64333\Desktop\zou_gray.jpg"  # 替换为你的图像路径
output_path = r'compressed_image.jpg'
compression_ratio = wavelet_compress_image(image_path, output_path, wavelet='db1', level=5, threshold=100)
print(f"Compression Ratio: {compression_ratio:.2f}%")