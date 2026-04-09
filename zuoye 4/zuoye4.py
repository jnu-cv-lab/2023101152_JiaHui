import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, fft

plt.switch_backend('Agg')

#  选取测试信号 
print("=" * 70)
print("Part 3: 选取测试信号")
print("=" * 70)
x = np.array([100, 90, 80, 70, 60, 50, 40, 30], dtype=float)
N = len(x)
print("原始一维信号 x[n] (模拟图像一行像素):", x)
print("信号长度 N =", N)

# 延拓方式对比 
print("\n" + "=" * 70)
print("Part 4: 延拓方式对比")
print("=" * 70)

# 4.1 DFT 隐含的周期延拓（展示2个周期）
periodic_ext = np.tile(x, 2)  # 周期重复
print("\n【DFT周期延拓】2个周期后的序列:")
print(periodic_ext)
print("边界差异说明：")
print(f"  原始信号末尾 = {x[-1]}, 下一个周期开头 = {x[0]}")
print(f"  在连接处 {x[-1]} → {x[0]} 存在数值跳变，导致高频分量产生。")

# 4.2 DCT 隐含的偶对称延拓（标准镜像，保留端点）
even_ext = np.concatenate([x, x[::-1]])  # 长度为 2N
print("\n【DCT偶对称延拓】镜像后的序列:")
print(even_ext)
print("边界差异说明：")
print(f"  原始信号末尾 = {x[-1]}, 镜像部分开头 = {x[-1]} (相同)")
print(f"  连接处连续无跳变，且呈偶对称，边界光滑，因此高频成分少。")
# 保存图像
fig1 = plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.stem(range(N), x, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title('Original Signal')
plt.xlabel('Sample Index n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.stem(range(2 * N), periodic_ext, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title('DFT Periodic Extension (Boundary Jump)')
plt.xlabel('Sample Index n')
plt.grid(True)
plt.axvline(x=N-0.5, color='gray', linestyle='--', linewidth=1)
plt.text(N-1, max(periodic_ext)*0.8, 'Jump', ha='center', color='red')

plt.subplot(1, 3, 3)
plt.stem(range(2 * N), even_ext, linefmt='g-', markerfmt='go', basefmt='k-')
plt.title('DCT Even Symmetric Extension (Smooth Boundary)')
plt.xlabel('Sample Index n')
plt.grid(True)
plt.axvline(x=N-0.5, color='gray', linestyle='--', linewidth=1)
plt.text(N-1, max(even_ext)*0.8, 'Smooth', ha='center', color='green')

plt.tight_layout()
plt.savefig('dft_dct_extension_comparison.png', dpi=150, bbox_inches='tight')
print("\n[Image saved] dft_dct_extension_comparison.png")
plt.close(fig1)

# 频谱对比与能量集中性
print("\n" + "=" * 70)
print("第5部分：频谱对比")
print("=" * 70)

# 5.1 计算 DFT（幅度谱）
X_dft = fft(x)
X_dft_mag = np.abs(X_dft)
print("\nDFT幅度谱 (前 N 个系数):")
for k in range(N):
    print(f"  k={k}: {X_dft_mag[k]:.2f}")

# 5.2 计算 DCT（DCT-II，标准定义）
X_dct = dct(x, type=2, norm=None)
print("\nDCT系数 (DCT-II):")
for k in range(N):
    print(f"  k={k}: {X_dct[k]:.2f}")

# 5.3 能量集中性比较（前若干个系数能量占比）
energy_dft = X_dft_mag ** 2
energy_dct = X_dct ** 2
total_dft = np.sum(energy_dft)
total_dct = np.sum(energy_dct)

print("\n【能量集中性对比】")
print("系数个数\tDFT累计能量占比\tDCT累计能量占比")
cum_dft = 0
cum_dct = 0
for i in range(N):
    cum_dft += energy_dft[i]
    cum_dct += energy_dct[i]
    ratio_dft = cum_dft / total_dft * 100
    ratio_dct = cum_dct / total_dct * 100
    print(f"前{i+1}个\t\t{ratio_dft:.2f}%\t\t\t{ratio_dct:.2f}%")


# Plot spectrum comparison and save
fig2 = plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(range(N), X_dft_mag, linefmt='r-', markerfmt='ro', basefmt='k-')
plt.title('DFT Magnitude Spectrum (with High-Frequency Leakage)')
plt.xlabel('Frequency Index k')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(range(N), X_dct, linefmt='b-', markerfmt='bo', basefmt='k-')
plt.title('DCT Coefficients (Low-Frequency Concentrated)')
plt.xlabel('Frequency Index k')
plt.ylabel('Coefficient Value')
plt.grid(True)

plt.tight_layout()
plt.savefig('dft_dct_spectrum_comparison.png', dpi=150, bbox_inches='tight')
print("\n[Image saved] dft_dct_spectrum_comparison.png")
plt.close(fig2)

print("\nProgram finished. All images have been saved to the current directory.")
