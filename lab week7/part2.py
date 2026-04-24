import cv2
import numpy as np

# ---------------------- 1. 读取图像 + ORB 特征检测----------------------
# 读取灰度图
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

# 创建 ORB 检测器
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# ---------------------- 2. 暴力匹配器 BFMatcher） ----------------------
# 创建匹配器：ORB 必须用 NORM_HAMMING，开启 crossCheck
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 执行匹配
matches = bf.match(des1, des2)

# 按匹配距离从小到大排序（距离越小，匹配越准确）
matches = sorted(matches, key=lambda x: x.distance)

# ---------------------- 3. 输出结果 ----------------------
print("===== ORB 特征匹配结果 =====")
print(f"总匹配数量：{len(matches)}")
#输出结果是287

# ---------------------- 4. 绘制并保存匹配图 ----------------------
# 绘制 全部 初始匹配图
img_matches_all = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# 绘制 前30个 最优匹配
img_matches_top30 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite('7_orb_all_matches.png', img_matches_all)    # 初始匹配全图
cv2.imwrite('7_orb_top30_matches.png', img_matches_top30) # 前30个最优匹配
