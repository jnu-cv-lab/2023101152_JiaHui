import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 自编：直方图均衡化
def my_equalize_hist(img):
    h, w = img.shape
    hist = np.zeros(256, dtype=np.int32)
    for y in range(h):
        for x in range(w):
            hist[img[y, x]] += 1
    total = h * w
    p = hist / total
    cdf = np.zeros_like(p)
    cdf[0] = p[0]
    for i in range(1, 256):
        cdf[i] = cdf[i-1] + p[i]
    s = np.uint8(255 * cdf)
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            out[y, x] = s[img[y, x]]
    return out

#  2. 生成低对比、噪声图像
def low_contrast(img, alpha=0.4, beta=20):
    return np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, sigma=15):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)

def add_salt_pepper(img, prob=0.02):
    out = img.copy()
    h, w = img.shape
    pts = np.random.randint(0, h*w, int(prob*h*w))
    out.flat[pts] = 255
    pts = np.random.randint(0, h*w, int(prob*h*w))
    out.flat[pts] = 0
    return out

# 3. 锐化（Laplacian）
def sharpen(img):
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return np.clip(img - 0.4*lap, 0, 255).astype(np.uint8)

# 4. 定量评价指标 
def entropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    hist = hist / hist.sum() + 1e-8
    return -np.sum(hist * np.log2(hist))

def average_gradient(img):
    gx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    gy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    return np.mean(np.sqrt(gx**2 + gy**2))

# 主流程 
if __name__ == "__main__":
    img = cv2.imread("/mnt/c/Users/Lenovo/Pictures/Camera Roll/test.xuebao.jpg", 0)
    if img is None:
        raise FileNotFoundError("图片读取失败！")

    # 生成三种问题图像
    img_low = low_contrast(img)
    img_gauss = add_gaussian_noise(img)
    img_sp = add_salt_pepper(img)

    imgs = [
        ("Original", img),
        ("LowContrast", img_low),
        ("GaussianNoise", img_gauss),
        ("SaltPepper", img_sp)
    ]

    # 绘图
    plt.figure(figsize=(28, 12))
    idx = 1
    for name, I in imgs:
        he = my_equalize_hist(I)
        clahe = cv2.createCLAHE(clipLimit=2).apply(I)
        mean3 = cv2.blur(I, (3,3))
        gauss3 = cv2.GaussianBlur(I,(3,3),0)
        median = cv2.medianBlur(I,3)
        sharp = sharpen(I)
        comb1 = my_equalize_hist(gauss3)
        comb2 = gauss3

        results = [(name, I), ("HE", he), ("CLAHE", clahe),
                   ("Mean3", mean3), ("Gauss3", gauss3), ("Median", median),
                   ("Sharpen", sharp), ("Filter→HE", comb1), ("HE→Filter", comb2)]

        for title, res in results:
            plt.subplot(len(imgs), 18, idx)
            plt.imshow(res, cmap="gray")
            plt.title(title, fontsize=8)
            plt.axis("off")
            idx += 1
            plt.subplot(len(imgs), 18, idx)
            plt.hist(res.flatten(), bins=256, range=(0,256), color="k", lw=0.3)
            plt.axis("off")
            idx += 1

    plt.tight_layout()
    plt.savefig("result_all.png", dpi=200)
    print("✅ 已生成对比图：result_all.png")

    # 输出所有处理方法的定量指标
    print("\n===== 定量指标对比（处理前 vs 处理后） =====")
    for name, I in imgs:
        he = my_equalize_hist(I)
        clahe = cv2.createCLAHE(clipLimit=2).apply(I)
        mean3 = cv2.blur(I, (3,3))
        gauss3 = cv2.GaussianBlur(I,(3,3),0)
        median = cv2.medianBlur(I,3)
        sharp = sharpen(I)
        comb1 = my_equalize_hist(gauss3)
        comb2 = gauss3

        print(f"\n🔹 {name}")
        for method, res in [
            ("Original", I), ("HE", he), ("CLAHE", clahe),
            ("Mean3", mean3), ("Gauss3", gauss3), ("Median", median),
            ("Sharoen", sharp), ("Filter→HE", comb1), ("HE→Filter", comb2)
        ]:
            e = entropy(res)
            g = average_gradient(res)
            print(f"  {method:<8}  熵={e:.2f}   梯度={g:.1f}")
