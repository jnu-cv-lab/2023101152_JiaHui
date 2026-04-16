import numpy as np
import cv2
import matplotlib.pyplot as plt

# ===================== 参数 =====================
BLOCK_SIZE = 32
ENERGY_THRESH = 0.95
IMAGE_PATH = "/home/mjhyyfj/cv-course/model.jpg"
# ===================== 读取图像 =====================
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
H, W = gray.shape
gray = gray[:(H//BLOCK_SIZE)*BLOCK_SIZE, :(W//BLOCK_SIZE)*BLOCK_SIZE]
print("图像尺寸:", gray.shape)

# ===================== FFT 95% 能量最高频率 =====================
def fft_95p_max_freq(block):
    fft = np.fft.fft2(block)
    fft_shift = np.fft.fftshift(fft)
    energy = np.abs(fft_shift) ** 2
    total_energy = np.sum(energy)
    
    # 生成频率坐标
    y_freq, x_freq = np.meshgrid(np.arange(-BLOCK_SIZE//2, BLOCK_SIZE//2),
                                 np.arange(-BLOCK_SIZE//2, BLOCK_SIZE//2),
                                 indexing='ij')
    freq_map = np.sqrt(x_freq**2 + y_freq**2)  # 径向频率
    
    # 按频率从小到大排序（作业要求！）
    sorted_pairs = sorted(zip(freq_map.flatten(), energy.flatten()), key=lambda x: x[0])
    freq_sorted = np.array([p[0] for p in sorted_pairs])
    energy_sorted = np.array([p[1] for p in sorted_pairs])
    
    # 累加能量直到 95%
    cum_energy = np.cumsum(energy_sorted) / total_energy
    max_freq_idx = np.argmax(cum_energy >= ENERGY_THRESH)
    max_freq = freq_sorted[max_freq_idx]
    
    # 归一化
    return max_freq / (BLOCK_SIZE // 2)

# ===================== 【核心：纯空域梯度能量法】 =====================
def spatial_95p_gradient_freq(block):
    h, w = block.shape

    # 1. 中央差分（比Sobel更精确的空域差分）
    dx = np.zeros_like(block)
    dy = np.zeros_like(block)

    # X 方向中央差分
    dx[:, 1:-1] = (block[:, 2:] - block[:, :-2]) / 2.0
    dx[:, 0] = block[:, 1] - block[:, 0]
    dx[:, -1] = block[:, -1] - block[:, -2]

    # Y 方向中央差分
    dy[1:-1, :] = (block[2:, :] - block[:-2, :]) / 2.0
    dy[0, :] = block[1, :] - block[0, :]
    dy[-1, :] = block[-1, :] - block[-2, :]

    # 2. 梯度幅值 & 梯度能量（平方）
    grad_mag = np.sqrt(dx**2 + dy**2)
    grad_energy = grad_mag ** 2  # 与FFT能量定义对齐

    # 3. 取 95% 梯度能量对应的最大梯度
    total_grad_energy = np.sum(grad_energy)
    flat_g = grad_energy.flatten()
    sorted_g = np.sort(flat_g)[::-1]

    cum = 0
    idx = 0
    for i, g in enumerate(sorted_g):
        cum += g
        if cum >= total_grad_energy * ENERGY_THRESH:
            idx = i
            break
    min_g = sorted_g[idx]

    # 4. 找到 >= 该能量的所有梯度，取最大幅值
    valid_grad = grad_mag[grad_energy >= min_g]
    max_grad = np.max(valid_grad) if len(valid_grad) > 0 else 0

    # 5. 映射为归一化频率（与FFT同范围 0~1）
    max_possible_grad = 255.0 * np.sqrt(2)  # 理论最大梯度
    freq_est = max_grad / max_possible_grad
    return np.clip(freq_est, 0, 1)

# ===================== 分块计算 =====================
fft_freqs = []
spatial_freqs = []

for y in range(0, gray.shape[0], BLOCK_SIZE):
    for x in range(0, gray.shape[1], BLOCK_SIZE):
        block = gray[y:y+BLOCK_SIZE, x:x+BLOCK_SIZE]
        f = fft_95p_max_freq(block)
        s = spatial_95p_gradient_freq(block)
        fft_freqs.append(f)
        spatial_freqs.append(s)

fft_arr = np.array(fft_freqs)
spa_arr = np.array(spatial_freqs)

# ===================== 评价指标 =====================
corr = np.corrcoef(fft_arr, spa_arr)[0, 1]
mae = np.mean(np.abs(fft_arr - spa_arr))
mse = np.mean((fft_arr - spa_arr)**2)

print("===== 纯空域梯度能量法结果 =====")
print(f"相关系数 corr:  {corr:.4f}")
print(f"平均绝对误差 MAE: {mae:.4f}")
print(f"均方误差 MSE:    {mse:.4f}")

# ===================== 绘图并保存到当前目录 =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

num_h = gray.shape[0] // BLOCK_SIZE
num_w = gray.shape[1] // BLOCK_SIZE

fft_map = fft_arr.reshape(num_h, num_w)
spa_map = spa_arr.reshape(num_h, num_w)

# 排版：2行，第一行3张，第二行1张
plt.figure(figsize=(18, 10))

# 第1行：3张图
# 原始图
plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Gray Image')
plt.axis('off')

# FFT频率图
plt.subplot(2, 3, 2)
plt.imshow(fft_map, cmap='jet')
plt.colorbar(label='Normalized Frequency')
plt.title('FFT 95% Energy Max Frequency')
plt.axis('off')

# 梯度频率图
plt.subplot(2, 3, 3)
plt.imshow(spa_map, cmap='jet')
plt.colorbar(label='Normalized Frequency')
plt.title('Gradient 95% Energy Frequency')
plt.axis('off')

# 第2行：散点图（居中放大）
plt.subplot(2, 1, 2)
plt.scatter(fft_arr, spa_arr, s=15, alpha=0.7, color='#4285F4')
plt.plot([0,1], [0,1], 'r--', lw=2, label='y=x (ideal)')
plt.xlabel('FFT Frequency')
plt.ylabel('Gradient Frequency')
plt.title(f'Corr = {corr:.4f} | MAE = {mae:.4f} | MSE = {mse:.4f}')
plt.grid(alpha=0.3)
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.tight_layout()

# 保存
plt.savefig("freq_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("对比图已保存至当前目录：freq_comparison.png")
