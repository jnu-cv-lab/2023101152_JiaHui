import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局设置 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 下采样函数 =====================
def downsample_direct(img, scale):
    """
    直接下采样（无预滤波）
    缺点：不滤波直接缩小，会产生高频混叠（锯齿）
    """
    h, w = img.shape
    new_h, new_w = int(h / scale), int(w / scale)
    # 双线性插值缩小图像
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return downsampled

def downsample_with_gaussian(img, scale):
    """
    高斯滤波 + 下采样
    优点：先平滑滤除高频，再缩小，可有效抑制混叠
    """
    h, w = img.shape
    new_h, new_w = int(h / scale), int(w / scale)
    # 高斯滤波（去除高频噪声/细节）
    smoothed = cv2.GaussianBlur(img, (5, 5), 1)
    # 下采样
    downsampled = cv2.resize(smoothed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return downsampled, smoothed

# ===================== 图像恢复（插值放大）函数 =====================
def restore_image(small_img, original_size, method):
    """
    使用不同插值方法将小图恢复为原图尺寸
    支持：最近邻、双线性、双三次
    """
    h, w = original_size
    if method == "Nearest":
        restored = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_NEAREST)
    elif method == "Bilinear":
        restored = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_LINEAR)
    elif method == "Bicubic":
        restored = cv2.resize(small_img, (w, h), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError("不支持的插值方法")
    return restored

# ===================== 质量评价指标 =====================
def calculate_mse_psnr(img1, img2):
    """
    计算图像的MSE（均方误差）和PSNR（峰值信噪比）
    PSNR越高，图像恢复质量越好
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    return mse, psnr

# ===================== FFT 傅里叶变换分析 =====================
def compute_fft_spectrum(img):
    """
    计算二维FFT频谱
    步骤：FFT变换 → 移频中心 → 取幅度 → 对数增强显示
    """
    gray = img.copy().astype(np.float32)
    fft = np.fft.fft2(gray)          # 二维离散傅里叶变换
    fft_shift = np.fft.fftshift(fft) # 将零频移到图像中心
    mag = np.abs(fft_shift)          # 幅度谱
    log_mag = 20 * np.log10(mag + 1e-8)# 对数缩放，便于观察
    return log_mag

def show_spectrum_comparison(original, direct_down, gauss_down, restored_bilinear, scale):
    """
    显示FFT频谱对比图
    包含：原图、直接下采样、高斯下采样、双线性恢复图的频谱
    """
    spec_ori = compute_fft_spectrum(original)
    spec_dd = compute_fft_spectrum(direct_down)
    spec_gd = compute_fft_spectrum(gauss_down)
    spec_rest = compute_fft_spectrum(restored_bilinear)

    plt.figure(figsize=(16, 10))
    plt.subplot(2,2,1)
    plt.imshow(spec_ori, cmap='jet')
    plt.title('Original\nSpectrum (Log Magnitude)')
    plt.axis('off')
    
    plt.subplot(2,2,2)
    plt.imshow(spec_dd, cmap='jet')
    plt.title(f'Direct Down 1/{scale}\nSpectrum')
    plt.axis('off')
    
    plt.subplot(2,2,3)
    plt.imshow(spec_gd, cmap='jet')
    plt.title(f'Gaussian Down 1/{scale}\nSpectrum')
    plt.axis('off')
    
    plt.subplot(2,2,4)
    plt.imshow(spec_rest, cmap='jet')
    plt.title(f'Bilinear Restored\nSpectrum')
    plt.axis('off')

    plt.suptitle(f'FFT Spectrum Comparison (Scale 1/{scale})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'FFT_Spectrum_Scale_{scale}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ===================== DCT 离散余弦变换分析 =====================
def compute_dct(img):
    """
    计算二维DCT（离散余弦变换）
    DCT常用于图像压缩，能量集中在左上角低频区
    """
    img_float = img.astype(np.float32)
    dct = cv2.dct(img_float)          # 二维DCT变换
    log_dct = np.log(np.abs(dct) + 1e-8)# 对数显示DCT系数
    return dct, log_dct

def calculate_low_freq_energy_ratio(dct_coeff, ratio=0.1):
    """
    计算左上角低频区域能量占总能量的比例
    低频区域：图像左上角 10% 大小区域
    能量 = 系数平方和
    """
    h, w = dct_coeff.shape
    low_h, low_w = int(h * ratio), int(w * ratio)
    
    # 低频能量（左上角）
    low_energy = np.sum(np.abs(dct_coeff[:low_h, :low_w]) ** 2)
    # 总能量
    total_energy = np.sum(np.abs(dct_coeff) ** 2)
    
    return low_energy / (total_energy + 1e-10), low_h, low_w

def show_dct_comparison(original, restored_dict, scale, down_type="Direct"):
    """
    显示DCT对比图：原图 vs 三种插值恢复图像
    展示DCT系数图 + 低频能量占比
    """
    ori_dct, ori_log = compute_dct(original)
    ori_ratio, lh, lw = calculate_low_freq_energy_ratio(ori_dct)
    
    methods = ["Nearest", "Bilinear", "Bicubic"]
    dct_results = [compute_dct(restored_dict[m]) for m in methods]
    ratios = [calculate_low_freq_energy_ratio(dct)[0] for dct, _ in dct_results]

    plt.figure(figsize=(18, 10))
    # 原图DCT
    plt.subplot(2,4,1)
    plt.imshow(ori_log, cmap='viridis')
    plt.title(f'Original DCT\nLow freq ratio: {ori_ratio:.3f}')
    plt.axis('off')

    # 三种恢复方法的DCT
    for i, (m, (dct, log_dct), r) in enumerate(zip(methods, dct_results, ratios)):
        plt.subplot(2,4,2+i)
        plt.imshow(log_dct, cmap='viridis')
        plt.title(f'{m}\nRatio: {r:.3f}')
        plt.axis('off')

    # 空间域原图
    plt.subplot(2,4,5)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 空间域恢复图
    for i, (m, im) in enumerate(zip(methods, [restored_dict[m] for m in methods])):
        plt.subplot(2,4,6+i)
        plt.imshow(im, cmap='gray')
        plt.title(f'{m} Restored')
        plt.axis('off')

    plt.suptitle(f'DCT Analysis - {down_type} Downsampling 1/{scale}\nLow-freq region: {lh}x{lw}', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'DCT_{down_type}_Scale_{scale}.png', dpi=150, bbox_inches='tight')
    plt.show()

# ===================== 主函数 =====================
def main():
    print("Reading grayscale image...")
    # 读取灰度图
    img = cv2.imread('lab4.jpg', cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("Error: lab4.jpg not found")
        return
    
    original_size = img.shape
    print(f"Original size: {original_size}")
    
    # 下采样倍数：2倍、4倍
    scales = [2, 4]
    # 插值方法
    methods = ["Nearest", "Bilinear", "Bicubic"]
    
    # 存储所有结果
    all_direct = {}
    all_gauss = {}
    
    for s in scales:
        print(f"\nScale factor: 1/{s}")
        # 1. 两种下采样方式
        dd = downsample_direct(img, s)       # 直接下采样
        gd, _ = downsample_with_gaussian(img, s)# 高斯下采样
        
        # 2. 三种插值方法恢复图像
        rest_d = {m: restore_image(dd, original_size, m) for m in methods}
        rest_g = {m: restore_image(gd, original_size, m) for m in methods}
        
        all_direct[s] = rest_d
        all_gauss[s] = rest_g
        
        # 3. 输出质量指标MSE/PSNR
        print("\nDirect downsampling + restoration:")
        for m in methods:
            mse, psnr = calculate_mse_psnr(img, rest_d[m])
            print(f"  {m}: MSE={mse:.2f}, PSNR={psnr:.2f}dB")
        
        print("\nGaussian downsampling + restoration:")
        for m in methods:
            mse, psnr = calculate_mse_psnr(img, rest_g[m])
            print(f"  {m}: MSE={mse:.2f}, PSNR={psnr:.2f}dB")
        
        # 4. 显示空间域对比图
        plt.figure(figsize=(18, 10))
        plt.subplot(3,5,1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Original\n{img.shape}')
        plt.axis('off')
        
        plt.subplot(3,5,2)
        plt.imshow(dd, cmap='gray')
        plt.title(f'Direct 1/{s}\n{dd.shape}')
        plt.axis('off')
        
        for i, m in enumerate(methods):
            im = rest_d[m]
            mse, psnr = calculate_mse_psnr(img, im)
            plt.subplot(3,5,3+i)
            plt.imshow(im, cmap='gray')
            plt.title(f'{m}\nMSE={mse:.1f}\nPSNR={psnr:.1f}')
            plt.axis('off')
        
        plt.subplot(3,5,5+2)
        plt.imshow(gd, cmap='gray')
        plt.title(f'Gaussian 1/{s}\n{gd.shape}')
        plt.axis('off')
        
        for i, m in enumerate(methods):
            im = rest_g[m]
            mse, psnr = calculate_mse_psnr(img, im)
            plt.subplot(3,5,5+3+i)
            plt.imshow(im, cmap='gray')
            plt.title(f'{m}\nMSE={mse:.1f}\nPSNR={psnr:.1f}')
            plt.axis('off')
        
        plt.suptitle(f'Downsampling & Interpolation Comparison (Scale 1/{s})', fontsize=16)
        plt.tight_layout()
        plt.show()

        # 5. 显示FFT频谱分析
        show_spectrum_comparison(img, dd, gd, rest_d["Bilinear"], s)

        # 6. 显示DCT变换分析
        show_dct_comparison(img, rest_d, s, down_type="Direct")
        show_dct_comparison(img, rest_g, s, down_type="Gaussian")

    # ===================== 保存3张综合大图 =====================
    print("\nGenerating and saving 3 comprehensive figures...")

    # 图1：所有方法综合对比图
    plt.figure(figsize=(20, 12))
    plt.suptitle('Overall Comparison: All Methods', fontsize=20)
    plt.subplot(3,6,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    for i, m in enumerate(methods):
        im = all_direct[2][m]
        mse, psnr = calculate_mse_psnr(img, im)
        plt.subplot(3,6,2+i)
        plt.imshow(im, cmap='gray')
        plt.title(f'Direct 1/2 + {m}\nMSE={mse:.1f}')
        plt.axis('off')
    
    for i, m in enumerate(methods):
        im = all_direct[4][m]
        mse, psnr = calculate_mse_psnr(img, im)
        plt.subplot(3,6,7+i)
        plt.imshow(im, cmap='gray')
        plt.title(f'Direct 1/4 + {m}\nMSE={mse:.1f}')
        plt.axis('off')
    
    for i, m in enumerate(methods):
        im = all_gauss[2][m]
        mse, psnr = calculate_mse_psnr(img, im)
        plt.subplot(3,6,13+i)
        plt.imshow(im, cmap='gray')
        plt.title(f'Gaussian 1/2 + {m}\nMSE={mse:.1f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('01_Overall_Comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图2：直接下采样恢复对比图
    plt.figure(figsize=(18,10))
    plt.suptitle('Direct Downsampling Restoration', fontsize=18)
    for i,m in enumerate(methods):
        im = all_direct[2][m]
        mse,psnr = calculate_mse_psnr(img,im)
        plt.subplot(2,3,1+i)
        plt.imshow(im,cmap='gray')
        plt.title(f'1/2 + {m}\nPSNR={psnr:.1f}')
        plt.axis('off')
    for i,m in enumerate(methods):
        im = all_direct[4][m]
        mse,psnr = calculate_mse_psnr(img,im)
        plt.subplot(2,3,4+i)
        plt.imshow(im,cmap='gray')
        plt.title(f'1/4 + {m}\nPSNR={psnr:.1f}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('02_Direct_Downsampling_Comparison.png',dpi=150,bbox_inches='tight')
    plt.close()

    # 图3：高斯下采样恢复对比图
    plt.figure(figsize=(18,10))
    plt.suptitle('Gaussian Downsampling Restoration', fontsize=18)
    for i,m in enumerate(methods):
        im = all_gauss[2][m]
        mse,psnr = calculate_mse_psnr(img,im)
        plt.subplot(2,3,1+i)
        plt.imshow(im,cmap='gray')
        plt.title(f'1/2 + {m}\nPSNR={psnr:.1f}')
        plt.axis('off')
    for i,m in enumerate(methods):
        im = all_gauss[4][m]
        mse,psnr = calculate_mse_psnr(img,im)
        plt.subplot(2,3,4+i)
        plt.imshow(im,cmap='gray')
        plt.title(f'1/4 + {m}\nPSNR={psnr:.1f}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('03_Gaussian_Downsampling_Comparison.png',dpi=150,bbox_inches='tight')
    plt.close()

    print("All figures saved successfully!")
    
    # ===================== 实验结论 =====================
    print("\n======== 实验结论 ========")
    print("1. 双三次插值（Bicubic）的PSNR最高，恢复质量最好")
    print("2. 高斯下采样能抑制混叠，但会丢失高频细节")
    print("3. 下采样倍数越大，高频损失越严重，恢复效果越差")
    print("\n======== FFT频谱分析 ========")
    print("1. 直接下采样保留高频 → 产生混叠")
    print("2. 高斯下采样滤除高频 → 无混叠但模糊")
    print("3. 插值恢复无法重建丢失的高频")
    print("\n======== DCT分析 ========")
    print("1. 原图高低频能量均衡")
    print("2. 恢复图像低频能量占比显著上升（高频丢失）")
    print("3. 双三次插值保留高频最多，效果最优")

if __name__ == "__main__":
    main()
