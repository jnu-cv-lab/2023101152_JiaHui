import cv2
import numpy as np

# 存储选点
points = []

def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"已选点 {len(points)}: ({x}, {y})")
        # 画点
        cv2.circle(img_show, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("select 4 corners: TL-TR-BR-BL", img_show)

#  读取图像
img = cv2.imread("test6.2.jpg")

img_show = img.copy()
h, w = img.shape[:2]

# 鼠标选四个点 
cv2.namedWindow("select 4 corners: TL-TR-BR-BL")
cv2.setMouseCallback("select 4 corners: TL-TR-BR-BL", mouse_click)

print("按顺序点击：左上 → 右上 → 右下 → 左下")
print("选完4个点后按任意键继续")

while True:
    cv2.imshow("select 4 corners: TL-TR-BR-BL", img_show)
    key = cv2.waitKey(1) & 0xFF
    if key != 255 or len(points) >= 4:
        break
cv2.destroyAllWindows()

if len(points) < 4:
    print("点数不足4个")
    exit()

src = np.float32(points[:4])

#  计算目标矩形 
# 输出文档宽高
width = int(max(
    np.linalg.norm(src[0] - src[1]),
    np.linalg.norm(src[2] - src[3])
))
height = int(max(
    np.linalg.norm(src[1] - src[2]),
    np.linalg.norm(src[3] - src[0])
))

dst = np.float32([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
])

# ===================== 1. 透视变换 =====================
M_persp = cv2.getPerspectiveTransform(src, dst)
warp = cv2.warpPerspective(img, M_persp, (width, height))

# ===================== 2. 相似变换（仿射实现） =====================
M_similar = cv2.getRotationMatrix2D((width//2, height//2), 0, 1.0)
similar = cv2.warpAffine(warp, M_similar, (width, height))

# ===================== 3. 仿射变换（3点） =====================
src_aff = src[:3]
dst_aff = dst[:3]
M_aff = cv2.getAffineTransform(src_aff, dst_aff)
affine = cv2.warpAffine(img, M_aff, (w, h))

#绘图
size = (400, 500)
img_r = cv2.resize(img, size)
aff_r = cv2.resize(affine, size)
warp_r = cv2.resize(similar, size)

result = np.hstack([img_r, aff_r, warp_r])

cv2.imwrite("6_document_corrected.jpg", result)
print("已保存：6_document_corrected.jpg")
print("从左到右：原图 | 仿射 | 透视+相似矫正")
