import cv2
import numpy as np

def orb_experiment(nfeatures):
    img1 = cv2.imread('box.png')
    img2 = cv2.imread('box_in_scene.png')
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    total_matches = len(matches)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    num_inliers = np.sum(mask)
    inlier_ratio = num_inliers / total_matches if total_matches > 0 else 0

    success = "否"  # 默认失败
    try:
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)
        dst_int = np.int32(dst)

        #  定位成功判断：4个角点都在场景图范围内，才判定成功
        x_coords = dst_int[:, 0, 0]
        y_coords = dst_int[:, 0, 1]
        x_in = np.all((x_coords >= 0) & (x_coords < w2))
        y_in = np.all((y_coords >= 0) & (y_coords < h2))

        if x_in and y_in:
            success = "是"
        else:
            success = "否"

        # 无论成功失败，都画出框
        img_result = img2.copy()
        cv2.polylines(img_result, [dst_int], True, (0, 0, 255), 3)
        cv2.imwrite(f'7_result_n{nfeatures}.png', img_result)

    except:
        success = "否"

    return [nfeatures, len(kp1), len(kp2), total_matches, num_inliers, round(inlier_ratio, 4), success]

# 运行实验
params_list = [500, 1000, 2000]
results = []
for p in params_list:
    res = orb_experiment(p)
    results.append(res)

# 输出表格
print("="*85)
print("                          ORB 参数对比实验结
