import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def downsample(img, factor):
    h, w = img.shape
    return img[:h//factor*factor, :w//factor*factor].reshape(h//factor, factor, w//factor, factor).mean(axis=(1,3))

# Part 1 
print("Part 1: Aliasing and Anti-aliasing")

size = 50
checker = (np.indices((size, size)).sum(0) % 2) * 1.0

# Chirp 信号范围 [-1,1]
x = np.linspace(-1, 1, size)
X, Y = np.meshgrid(x, x)
chirp = np.sin(12 * np.pi * (X**2 + Y**2))

factor = 4
checker_direct = downsample(checker, factor)
chirp_direct = downsample(chirp, factor)

# 抗混叠滤波
sigma_aa = 1.8
checker_aa = downsample(gaussian_filter(checker, sigma=sigma_aa), factor)
chirp_aa = downsample(gaussian_filter(chirp, sigma=sigma_aa), factor)

def fft_mag(img):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(img))))

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
imgs = [checker, checker_direct, checker_aa, chirp, chirp_direct, chirp_aa]
titles = ['Checkerboard', 'Direct', 'Anti-aliased', 'Chirp', 'Direct', 'Anti-aliased']

for i, ax in enumerate(axes.flat[:6]):
    ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(titles[i])
    ax.axis('off')

axes[1,2].imshow(fft_mag(checker_direct), cmap='hot')
axes[1,2].set_title('FFT (aliased)')
axes[1,3].imshow(fft_mag(checker_aa), cmap='hot')
axes[1,3].set_title('FFT (clean)')

for ax in axes[1,2:]:
    ax.axis('off')

plt.tight_layout()
plt.savefig('lab5_part1.png', dpi=150)
plt.show()

# Part 2
print("\nPart 2: Sigma optimization (M=4)")

M = 4
test_img = checker.copy()
sigmas = [0.5, 1.0, 2.0, 4.0]
h, w = test_img.shape

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i, ax in enumerate(axes.flat[:6]):
    # 棋盘格用 0~1，chirp 用 -1~1
    if i < 3:
        ax.imshow(imgs[i], cmap='gray', vmin=0, vmax=1)
    else:
        ax.imshow(imgs[i], cmap='gray', vmin=-1, vmax=1)
    ax.set_title(titles[i])
    ax.axis('off')

plt.suptitle('Sigma effect on downsampling (M=4)')
plt.tight_layout()
plt.savefig('lab5_part2.png', dpi=150)
plt.show()

# 理论对比：M=1.8，σ=0.45
print("\nCompare theoretical: M=1.8, σ=0.45")
M_theory = 1.8
sigma_theory = 0.45
img_small = checker[:50, :50]  # 正确！
h2, w2 = img_small.shape

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# 直接下采样
ds_direct = img_small[::2, ::2]
axes[0].imshow(cv2.resize(ds_direct, (w2, h2)), cmap='gray', vmin=0, vmax=1)
axes[0].set_title('No filter (aliased)')

# 理论 σ=0.45
ds_theory = gaussian_filter(img_small, sigma_theory)[::2, ::2]
axes[1].imshow(cv2.resize(ds_theory, (w2, h2)), cmap='gray', vmin=0, vmax=1)
axes[1].set_title('Theoretical σ=0.45')

# σ=1.8
ds_big = gaussian_filter(img_small, 1.8)[::2, ::2]
axes[2].imshow(cv2.resize(ds_big, (w2, h2)), cmap='gray', vmin=0, vmax=1)
axes[2].set_title('σ=1.8 (over blur)')

for ax in axes:
    ax.axis('off')

plt.suptitle('Theoretical vs empirical sigma')
plt.tight_layout()
plt.savefig('lab5_part2_theory.png', dpi=150)
plt.show()

#  Part 3 
print("\nPart 3: Adaptive downsampling")

img = np.zeros((512, 512))
X, Y = np.meshgrid(np.linspace(-1, 1, 512), np.linspace(-1, 1, 512))
R = np.sqrt(X**2 + Y**2)
img[R < 0.3] = np.sin(30 * np.pi * X[R < 0.3] * Y[R < 0.3])
img[(R >= 0.3) & (R < 0.6)] = np.sin(10 * np.pi * X[(R >= 0.3) & (R < 0.6)])
img[R >= 0.6] = np.sin(3 * np.pi * X[R >= 0.6])
img[:256, :256] = (np.indices((256, 256)).sum(0) % 2) * 1.0
img = (img - img.min()) / (img.max() - img.min())

grad = np.abs(cv2.Sobel(img, cv2.CV_64F, 1, 0, 3)) + np.abs(cv2.Sobel(img, cv2.CV_64F, 0, 1, 3))
block = 16
M_map = np.zeros((512 // block, 512 // block))

for i in range(M_map.shape[0]):
    for j in range(M_map.shape[1]):
        g = grad[i*block:(i+1)*block, j*block:(j+1)*block].mean()
        M_map[i,j] = min(4.0, 1.0 + g * 8)

adaptive = np.zeros((128, 128))
for i in range(M_map.shape[0]):
    for j in range(M_map.shape[1]):
        block_img = img[i*block:(i+1)*block, j*block:(j+1)*block]
        sigma = 0.45 * M_map[i,j]
        filtered = gaussian_filter(block_img, sigma)
        adaptive[i*4:(i+1)*4, j*4:(j+1)*4] = filtered[::4, ::4]

uniform = downsample(gaussian_filter(img, 1.8), 4)

def upsample_err(original, downsampled):
    h, w = original.shape
    up = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.abs(original - up)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0,0].imshow(img, cmap='gray')
axes[0,0].set_title('Original')
axes[0,1].imshow(uniform, cmap='gray', vmin=0, vmax=1)
axes[0,1].set_title('Uniform (global σ)')
axes[0,2].imshow(adaptive, cmap='gray', vmin=0, vmax=1)
axes[0,2].set_title('Adaptive (local σ)')
axes[1,0].imshow(grad, cmap='hot')
axes[1,0].set_title('Gradient map')
axes[1,1].imshow(upsample_err(img, uniform), cmap='hot', vmin=0, vmax=0.3)
axes[1,1].set_title(f'Uniform error MSE={np.mean(upsample_err(img, uniform)**2):.4f}')
axes[1,2].imshow(upsample_err(img, adaptive), cmap='hot', vmin=0, vmax=0.3)
axes[1,2].set_title(f'Adaptive error MSE={np.mean(upsample_err(img, adaptive)**2):.4f}')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig('lab5_part3.png', dpi=150)
plt.show()

print("\n程序运行结束！")
