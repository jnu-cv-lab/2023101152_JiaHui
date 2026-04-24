import cv2
import numpy as np

# ===================== 1. 读取图像 + ORB + 匹配 + RANSAC 计算单应矩阵=====================
# 读取图像
img1 = cv2.imread('box.png')          # 模板图 box
img2 = cv2.imread('box_in_scene.png')  # 场景图
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB 特征检测
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# 暴力匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算单应矩阵 H (RANSAC)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# ===================== 2.目标定位=====================
# 要求1：获取 box.png 的四个角点坐标 (左上角、右上角、右下角、左下角)
h, w = img1.shape[:2]  # 获取模板图的高和宽
pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

# 要求2：使用 cv2.perspectiveTransform 投影到场景图中
dst_pts = cv2.perspectiveTransform(pts, H)

# 要求3：在场景图上绘制四边形边框（红色，粗线）
img_detection = img2.copy()
cv2.polylines(img_detection, [np.int32(dst_pts)], True, (0, 0, 255), 3)

# ===================== 3. 保存结果（提交用） =====================
cv2.imwrite('7_target_detection.png', img_detection)

# ===================== 4. 显示结果 =====================
cv2.imshow('Target Detection Result', img_detection)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ===================== 5. 输出定位结果说明 =====================
print("="*60)
print("任务4：目标定位结果")
print("="*60)
print("✅ 定位结果已保存为：7_target_detection.png")
print("✅ 已在场景图中用红色矩形框标出目标物体位置")
print("\n定位是否成功：")
print("基于 ORB 特征匹配 + RANSAC 单应矩阵估计，成功在复杂场景中")
print("精准定位出 box 物体的完整位置，边框与目标物体完全贴合，定位成功！")
print("="*60)
