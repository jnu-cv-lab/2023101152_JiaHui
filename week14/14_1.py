# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ===================== 全局配置（严格按照作业要求） =====================
# 1. WSL 数据集根路径 D:\baminton
DATA_ROOT = "/mnt/d/baminton"
# 5. 统一重采样帧数 T=30
TARGET_FRAMES = 30
# 4. 单帧维度：33关键点 × 4特征(x,y,z,visibility) = 132
FRAME_DIM = 132
# 7. 训练集/测试集划分比例
TEST_SIZE = 0.2
# 输出目录 + 文件前缀 14_
SAVE_DIR = "./14_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# 标签映射：与实际文件夹名（下划线）一一对应
LABEL_MAP = {
    0: "forehand_drive",
    1: "forehand_lift",
    2: "forehand_net_shot",
    3: "forehand_clear",
    4: "backhand_drive",
    5: "backhand_net_shot"
}
NAME_TO_ID = {name: idx for idx, name in LABEL_MAP.items()}

# 初始化 MediaPipe Pose（作业要求：提取人体33个关键点）
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===================== 工具函数（对应作业步骤2~6） =====================
def frame_to_keypoint_vector(frame):
    """
    步骤2+3+4：单帧图像 -> 提取33关键点 -> 展平为132维向量
    :param frame: OpenCV读取的单帧图像
    :return: 132维特征向量
    """
    # BGR转RGB（MediaPipe要求）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)

    # 未检测到人体，返回全0向量
    if not results.pose_landmarks:
        return np.zeros(FRAME_DIM, dtype=np.float32)

    # 遍历33个关键点，拼接 x,y,z,visibility
    keypoints = []
    for landmark in results.pose_landmarks.landmark:
        keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return np.array(keypoints, dtype=np.float32)


def resample_sequence(frames_feature_list, target_len):
    """
    步骤5：不等长帧序列 等间隔重采样为固定帧数 T=30
    :param frames_feature_list: 原始视频所有帧的特征列表
    :param target_len: 目标帧数 30
    :return: 重采样后的帧特征列表
    """
    ori_length = len(frames_feature_list)
    if ori_length == 0:
        return [np.zeros(FRAME_DIM)] * target_len
    # 等间隔采样
    sample_indices = np.linspace(0, ori_length - 1, target_len, dtype=int)
    return [frames_feature_list[i] for i in sample_indices]


def normalize_skeleton(seq):
    """
    步骤6：骨架归一化
    规则：以左右髋部中心为原点，以肩宽为尺度做归一化
    :param seq: [30, 132] 骨架序列
    :return: 归一化后 [30, 132] 序列
    """
    # 重塑维度: [帧数, 33关键点, 4特征]
    seq_reshape = seq.reshape(TARGET_FRAMES, 33, 4)

    # 左右髋部关键点索引：左髋23，右髋24
    left_hip = seq_reshape[:, 23, :2]
    right_hip = seq_reshape[:, 24, :2]
    hip_center = (left_hip + right_hip) / 2.0  # 髋部中心（原点）

    # 左右肩部关键点索引：左肩11，右肩12
    left_shoulder = seq_reshape[:, 11, :2]
    right_shoulder = seq_reshape[:, 12, :2]
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)
    # 防止除0
    shoulder_width[shoulder_width < 1e-6] = 1e-6

    # 平移 + 尺度缩放归一化
    seq_reshape[..., :2] = (seq_reshape[..., :2] - hip_center[:, None, :]) / shoulder_width[:, None, :]

    # 还原为 [30, 132]
    return seq_reshape.reshape(TARGET_FRAMES, FRAME_DIM)


def video_to_skeleton_seq(video_path):
    """
    整合：读取单个视频 -> 逐帧提取特征 -> 重采样 -> 归一化
    :param video_path: 视频完整路径
    :return: [30, 132] 骨架序列 / None（读取失败）
    """
    # 步骤2：OpenCV读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    frame_feature_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 步骤3+4：单帧转为132维向量
        feat = frame_to_keypoint_vector(frame)
        frame_feature_list.append(feat)
    cap.release()

    if len(frame_feature_list) == 0:
        return None

    # 步骤5：重采样为30帧
    resampled_feat = resample_sequence(frame_feature_list, TARGET_FRAMES)
    seq_arr = np.array(resampled_feat, dtype=np.float32)
    # 步骤6：骨架归一化
    norm_seq = normalize_skeleton(seq_arr)
    return norm_seq

# ===================== 主处理流程（步骤1 + 7） =====================
def main():
    print("=" * 60)
    print("开始执行羽毛球视频 -> 骨架序列 预处理")
    print(f"数据集路径: {DATA_ROOT}")
    print(f"统一帧数: {TARGET_FRAMES}, 单帧维度: {FRAME_DIM}")
    print("=" * 60)

    # 步骤1：遍历类别文件夹
    # 1.1 筛选纯文件夹，过滤 .gitattributes 等文件
    all_entries = os.listdir(DATA_ROOT)
    class_folders = []
    for entry in all_entries:
        full_path = os.path.join(DATA_ROOT, entry)
        if os.path.isdir(full_path) and entry in NAME_TO_ID:
            class_folders.append(entry)

    if len(class_folders) == 0:
        print("❌ 未找到有效类别文件夹，请检查文件夹名称！")
        return
    print(f"\n检测到有效类别文件夹: {class_folders}")

    all_sequences = []
    all_labels = []
    total_videos = 0
    failed_videos = 0

    # 遍历每个类别文件夹
    for folder_name in class_folders:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        label_id = NAME_TO_ID[folder_name]
        print(f"\n----- 正在处理类别: {folder_name} (标签ID: {label_id}) -----")

        # 筛选视频文件：mp4 / avi / mov / mkv
        video_suffix = (".mp4", ".avi", ".mov", ".mkv")
        video_list = [v for v in os.listdir(folder_path) if v.lower().endswith(video_suffix)]
        print(f"当前类别视频总数: {len(video_list)}")

        # 遍历当前类别所有视频
        for video_name in tqdm(video_list, desc=f"处理视频"):
            total_videos += 1
            video_full_path = os.path.join(folder_path, video_name)
            skeleton_seq = video_to_skeleton_seq(video_full_path)

            if skeleton_seq is not None:
                all_sequences.append(skeleton_seq)
                all_labels.append(label_id)
            else:
                failed_videos += 1

    # 统计汇总
    print("\n" + "=" * 60)
    print(f"总扫描视频数: {total_videos}")
    print(f"读取/提取失败视频数: {failed_videos}")
    print(f"有效骨架样本数: {len(all_sequences)}")

    if len(all_sequences) == 0:
        print("❌ 无有效样本，程序终止！")
        return

    # 转为numpy数组
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)
    print(f"原始数据集形状 X: {X.shape}, y: {y.shape}")

    # 步骤7：划分训练集、测试集 test_size=0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    print(f"训练集 X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"测试集  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 保存 .npy 文件（带 14_ 前缀）
    np.save(os.path.join(SAVE_DIR, "14_X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIR, "14_y_train.npy"), y_train)
    np.save(os.path.join(SAVE_DIR, "14_X_test.npy"), X_test)
    np.save(os.path.join(SAVE_DIR, "14_y_test.npy"), y_test)

    # 保存标签映射文件
    with open(os.path.join(SAVE_DIR, "14_label_map.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_MAP, f, ensure_ascii=False, indent=2)

    print("\n✅ 预处理全部完成！")
    print(f"所有文件已保存至: {SAVE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
    # 释放MediaPipe资源
    pose_model.close()
