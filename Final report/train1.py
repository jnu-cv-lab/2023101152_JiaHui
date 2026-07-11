#!/usr/bin/env python3
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# ========== 配置 ==========
BASE_DIR = "/home/minmin/cv-course/lab01/bighomework/"           # 你的项目根目录
DB_ROOT = os.path.join(BASE_DIR, "tttrain") # 人脸库根目录
FEATURES_FILE = os.path.join(BASE_DIR, "face_features.npz")  # 保存的特征文件
# ==========================

def register():
    # 初始化模型
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))

    features_list = []
    labels_list = []

    # 遍历 face_db 下的每个人
    for person_name in os.listdir(DB_ROOT):
        person_dir = os.path.join(DB_ROOT, person_name)
        if not os.path.isdir(person_dir):
            continue

        # 遍历该人物的所有图片
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ 无法读取图片：{img_path}")
                continue

            faces = app.get(img)
            if len(faces) == 0:
                print(f"⚠️ 未检测到人脸：{img_path}")
                continue

            # 取面积最大的人脸
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            embedding = best_face.normed_embedding  # 归一化特征向量

            features_list.append(embedding)
            labels_list.append(person_name)
            print(f"✅ 已注册：{person_name} ← {img_name}")

    if len(features_list) == 0:
        print("❌ 没有注册到任何人脸，请检查 face_db 中的图片。")
        return

    # 保存特征
    np.savez(FEATURES_FILE,
             features=np.array(features_list),
             labels=np.array(labels_list))
    print(f"\n特征库已保存至：{FEATURES_FILE}")
    print(f"共注册 {len(labels_list)} 张人脸（{len(set(labels_list))} 人）")


if __name__ == "__main__":
    register()