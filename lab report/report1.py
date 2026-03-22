import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 任务1: 读取图片 ----------------------
# WSL 中访问 Windows 路径的格式：/mnt/c/Users/...
image_path = '/mnt/c/Users/Lenovo/Pictures/Camera Roll/test.cat.jpg'
img = cv2.imread(image_path)

# 检查是否读取成功
if img is None:
    print("❌ 错误：无法读取图片！请检查路径是否正确。")
    exit()

# ---------------------- 任务2: 输出图像基本信息 ----------------------
print("=" * 30 + " 图像基本信息 " + "=" * 30)
height, width = img.shape[:2]
channels = img.shape[2] if len(img.shape) == 3 else 1
print(f"图像尺寸 (宽×高): {width} × {height}")
print(f"图像通道数: {channels}")
print(f"像素数据类型: {img.dtype}")
print(f"总像素数: {img.size}")

# ---------------------- 任务3: 显示原图 ----------------------
plt.figure("原图", figsize=(8, 6))
# OpenCV 读取为 BGR，Matplotlib 显示为 RGB，需要转换
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.tight_layout()
plt.show()

# ---------------------- 任务4: 转换为灰度图并显示 ----------------------
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure("灰度图", figsize=(8, 6))
plt.imshow(gray_img, cmap="gray")
plt.title("Grayscale Image")
plt.axis("off")
plt.tight_layout()
plt.show()

# ---------------------- 任务5: 保存处理结果 ----------------------
# 保存到当前实验目录下（lab01-image-basic）
gray_save_path = "grayscale_result.jpg"
cv2.imwrite(gray_save_path, gray_img)
print(f"✅ 灰度图已保存到: {gray_save_path}")

# ---------------------- 任务6: NumPy 简单操作 ----------------------
print("=" * 30 + " NumPy 像素操作 " + "=" * 30)

# 1. 输出指定坐标的像素值 (x=200, y=200)
x, y = 200, 200
pixel_bgr = img[y, x]
print(f"坐标 ({x}, {y}) 的 BGR 像素值: {pixel_bgr}")

# 2. 裁剪左上角 200×200 区域并保存
cropped_img = img[0:200, 0:200]
crop_save_path = "cropped_result.jpg"
cv2.imwrite(crop_save_path, cropped_img)
print(f"✅ 裁剪图已保存到: {crop_save_path}")

# 显示裁剪结果
plt.figure("裁剪结果", figsize=(6, 6))
cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
plt.imshow(cropped_img_rgb)
plt.title("Cropped Image (200×200)")
plt.axis("off")
plt.tight_layout()
plt.show()

print("=" * 30 + " 实验完成 " + "=" * 30)
