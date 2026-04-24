import cv2
import numpy as np

# ===================== 1. 读取图像 + ORB 特征检测 + 暴力匹配 =====================
# 读取灰度图像
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 初始化ORB检测器
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 初始化暴力匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# 总匹配数量
total_matches = len(matches)

# ===================== 2. 提取匹配点对坐标 =====================
# 存储两幅图像对应的特征点坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# ===================== 3. 计算单应矩阵 + RANSAC剔除误匹配 =====================
# 计算单应矩阵，使用RANSAC算法，重投影误差阈值5.0
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# mask是一个二值矩阵，1表示内点（正确匹配），0表示外点（错误匹配）
matches_mask = mask.ravel().tolist()

# ===================== 4. 统计内点数量和内点比例 =====================
num_inliers = np.sum(mask)  # 内点数量
inlier_ratio = num_inliers / total_matches  # 内点比例

# ===================== 5. 绘制RANSAC后的内点匹配图 =====================
# 绘制内点匹配结果
img_ransac = cv2.drawMatches(
    img1, kp1, img2, kp2, matches, None,
    matchColor=(0, 255, 0),  # 内点用绿色
    singlePointColor=None,
    matchesMask=matches_mask,  # 只显示mask标记的内点
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 保存图片
cv2.imwrite('7_ransac_matches.png', img_ransac)

# ===================== 6. 输出所有提交需要的结果 =====================
print("="*50)
print("任务3：RANSAC剔除错误匹配 结果")
print("="*50)
print(f"总匹配数量：{total_matches}")
print(f"RANSAC内点数量：{num_inliers}")
print(f"内点比例：{inlier_ratio:.4f}")
print("\nHomography 单应矩阵：")
print(np.round(H, 4))  
print("="*50)
