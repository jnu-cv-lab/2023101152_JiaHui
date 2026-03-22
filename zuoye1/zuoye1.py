import cv2
import numpy as np
import math

# 1. 读入图像
img_path = "/mnt/c/Users/Lenovo/Pictures/Camera Roll/test.picture.jpg"
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"图片读入失败！路径：{img_path}")

h, w = img.shape[:2]
print(f"✅ 图片读入成功：{h}×{w}")

# 2. 转换为 YCrCb 并拆分通道
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
Y, Cr, Cb = cv2.split(img_ycrcb)

# 3. 对 Cb/Cr 下采样（2倍）
scale = 2
Cb_down = Cb[::scale, ::scale]
Cr_down = Cr[::scale, ::scale]

# 4. 插值恢复原尺寸（双线性插值）
Cb_up = cv2.resize(Cb_down, (w, h), interpolation=cv2.INTER_LINEAR)
Cr_up = cv2.resize(Cr_down, (w, h), interpolation=cv2.INTER_LINEAR)

# 5. 重建图像并转回 BGR
img_ycrcb_recon = cv2.merge((Y, Cr_up, Cb_up))
img_recon = cv2.cvtColor(img_ycrcb_recon, cv2.COLOR_YCrCb2BGR)

# 6. 计算 PSNR
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return 20 * math.log10(max_pixel / math.sqrt(mse))

psnr = calculate_psnr(img, img_recon)
print(f"📊 PSNR 值：{psnr:.2f} dB")

# 7. 保存结果到 Windows 目录
save_path = "/mnt/c/Users/Lenovo/Pictures/Camera Roll/reconstructed.jpg"
cv2.imwrite(save_path, img_recon)
print(f"💾 重建图片已保存到：{save_path}")
