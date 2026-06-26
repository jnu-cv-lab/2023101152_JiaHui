# -*- coding: utf-8 -*-
import cv2
import numpy as np

# ========== 要测试的图片路径 ==========
test_img_path = "/home/mjhyyfj/cv-course/src/lab01-image-basic/calib_imgs/img13.jpg"

# 棋盘内角点规格：9列 × 6行
corner_w, corner_h = 9, 6

# 1. 读取图片
img = cv2.imread(test_img_path)
if img is None:
    print(f"错误：无法读取图片 {test_img_path}")
    exit()

# 2. 转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 检测棋盘角点
ret, corners = cv2.findChessboardCorners(gray, (corner_w, corner_h), None)

print(f"角点检测结果：ret = {ret}")

if ret:
    # 4. 亚像素优化角点
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # 5. 绘制角点并保存
    draw_img = img.copy()
    cv2.drawChessboardCorners(draw_img, (corner_w, corner_h), corners_sub, ret)
    cv2.imwrite("16_debug_single_test.png", draw_img)
    print("✅ 角点检测成功！已保存角点图：16_debug_single_test.png")
else:
    print("❌ 角点检测失败，请检查：")
    print("1. 棋盘是否完整出现在画面中？")
    print("2. 图片是否有严重反光/模糊？")
    print("3. 棋盘内角点数量是否为9×6？")

cv2.imshow("Single Test Result", draw_img if ret else img)
cv2.waitKey(0)
cv2.destroyAllWindows()
