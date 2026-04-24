import cv2
import numpy as np
import time

# ===================== 通用函数：ORB 检测 =====================
def run_orb():
    img1 = cv2.imread('box.png')
    img2 = cv2.imread('box_in_scene.png')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # ORB
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    total = len(matches)

    # 单应矩阵
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = int(np.sum(mask))
    ratio = inliers / total if total else 0

    # 定位是否成功（4个角都在图内）
    success = "否"
    try:
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst_p = cv2.perspectiveTransform(pts, H)
        x = dst_p[:,0,0]
        y = dst_p[:,0,1]
        if np.all(x>0) and np.all(x<w2) and np.all(y>0) and np.all(y<h2):
            success = "是"
    except: pass

    return ["ORB", total, inliers, round(ratio,4), success, "较快"]

# ===================== SIFT 实验=====================
def run_sift():
    img1 = cv2.imread('box.png')
    img2 = cv2.imread('box_in_scene.png')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 1. SIFT 创建
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 2. BFMatcher + NORM_L2
    bf = cv2.BFMatcher(cv2.NORM_L2)

    # 3. KNN matching k=2
    matches = bf.knnMatch(des1, des2, k=2)

    # 4. Lowe ratio test 0.75
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    total = len(good)

    # 5. RANSAC + Homography
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(np.sum(mask))
        ratio = inliers / total if total else 0
    else:
        H = None
        total = 0
        inliers = 0
        ratio = 0

    # 定位是否成功
    success = "否"
    try:
        pts = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        dst_p = cv2.perspectiveTransform(pts, H)
        x = dst_p[:,0,0]
        y = dst_p[:,0,1]
        if np.all(x>0) and np.all(x<w2) and np.all(y>0) and np.all(y<h2):
            success = "是"
    except: pass

    # 保存定位图
    try:
        res = img2.copy()
        cv2.polylines(res, [np.int32(dst_p)], True, (0,0,255), 3)
        cv2.imwrite('7_sift_result.png', res)
    except: pass

    return ["SIFT", total, inliers, round(ratio,4), success, "较慢"]

# ===================== 运行并输出对比表格 =====================
orb_res = run_orb()
sift_res = run_sift()

print("="*90)
print("                           ORB vs SIFT 对比实验表格")
print("="*90)
print(f"{'方法':<8}{'匹配数量':<10}{'RANSAC内点数':<12}{'内点比例':<10}{'是否成功定位':<12}{'运行速度':<10}")
print("-"*90)
print(f"{orb_res[0]:<8}{orb_res[1]:<10}{orb_res[2]:<12}{orb_res[3]:<10}{orb_res[4]:<12}{orb_res[5]:<10}")
print(f"{sift_res[0]:<8}{sift_res[1]:<10}{sift_res[2]:<12}{sift_res[3]:<10}{sift_res[4]:<12}{sift_res[5]:<10}")
print("="*90)
