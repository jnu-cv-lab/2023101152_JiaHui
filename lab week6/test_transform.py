import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片
img = cv2.imread("test6.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

# 1. 相似变换
theta = np.radians(30)    # 旋转30度
scale = 0.8              # 缩放0.8
cx, cy = w//2, h//2       # 旋转中心

# 相似变换矩阵
M_similar = np.array([
    [scale * np.cos(theta), -scale * np.sin(theta), cx * (1 - scale * np.cos(theta)) + cy * scale * np.sin(theta)],
    [scale * np.sin(theta),  scale * np.cos(theta), cy * (1 - scale * np.cos(theta)) - cx * scale * np.sin(theta)]
], dtype=np.float32)

similar_img = cv2.warpAffine(img_rgb, M_similar, (w, h))


# 2. 仿射变换
M_affine = np.array([
    [0.7,  0.3,  30],   # x方向变换
    [0.2,  0.8,  50]    # y方向变换
], dtype=np.float32)

affine_img = cv2.warpAffine(img_rgb, M_affine, (w, h))


# 3. 透视变换

M_perspective = np.array([
    [0.8,  0.1,   20],
    [0.2,  0.9,   30],
    [0.0001, 0.0006, 1]
], dtype=np.float32)

perspective_img = cv2.warpPerspective(img_rgb, M_perspective, (w, h))

#显示并保存
plt.figure(figsize=(15, 5))

plt.subplot(1,3,1)
plt.imshow(similar_img)
plt.title("Similar (Manual Matrix)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(affine_img)
plt.title("Affine (Manual Matrix)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(perspective_img)
plt.title("Perspective (Manual Matrix)")
plt.axis("off")

plt.savefig("6_transform_manual.jpg", dpi=300, bbox_inches="tight")
plt.close()

print("6_transform_manual.jpg")
