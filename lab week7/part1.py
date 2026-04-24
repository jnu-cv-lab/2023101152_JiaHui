import cv2
import numpy as np

# 1. 读取两幅图像
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 2. 创建 ORB 检测器，设置 nfeatures=1000
orb = cv2.ORB_create(nfeatures=1000)

# 3. 检测关键点 + 计算描述子
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 4. 可视化关键点
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 保存可视化图片
cv2.imwrite('7_box_keypoints.png', img1_kp)
cv2.imwrite('7_box_in_scene_keypoints.png', img2_kp)

# 5. 输出关键点数量
print("===== 关键点数量 =====")
print(f"7_box.png 关键点数量：{len(kp1)}")
print(f"7_box_in_scene.png 关键点数量：{len(kp2)}")

# 6. 输出描述子维度
print("\n===== 描述子维度 =====")
print(f"box.png 描述子形状：{des1.shape}，维度：{des1.shape[1]}")
print(f"box_in_scene.png 描述子形状：{des2.shape}，维度：{des2.shape[1]}")
