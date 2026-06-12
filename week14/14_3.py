# -*- coding: utf-8 -*-
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== 全局配置（和训练/预处理完全对齐） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./14_data"
MODEL_PATH = "./14_skeleton_transformer.pth"

# 模型超参数（必须和训练代码一模一样）
INPUT_DIM = 132
SEQ_LEN = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1

# 推理视频路径：修改为你自己WSL下的视频路径
# 示例 "/mnt/d/baminton/forehand_clear/xxx.mp4"
TEST_VIDEO_PATH = r"/mnt/d/baminton/forehand_clear/004.mp4"

# 【可选项-可视化相关全局初始化】MediaPipe绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose_module = mp.solutions.pose

# 加载类别映射
with open(os.path.join(DATA_DIR, "14_label_map.json"), "r", encoding="utf-8") as f:
    LABEL_MAP = json.load(f)
# 数字ID -> 类别名称
ID_TO_NAME = {int(k): v for k, v in LABEL_MAP.items()}
CLASS_NAMES = list(LABEL_MAP.values())

# ===================== 1. 复用模型结构 =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.float)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[0, :, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class SkeletonTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_classes, dropout):
        super().__init__()
        self.linear_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        out = self.classifier(x)
        return out

# ===================== 2. 测试集Dataset（复用） =====================
class BadmintonSkeletonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = np.load(label_path)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq, label

# ===================== 3. 推理预处理工具 =====================
# 正确初始化Pose
pose = mp_pose_module.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def frame_to_vec(frame):
    """单帧提取132维向量"""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    if not res.pose_landmarks:
        return np.zeros(INPUT_DIM, dtype=np.float32)
    feat = []
    for lm in res.pose_landmarks.landmark:
        feat.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(feat, dtype=np.float32)

def resample_seq(feat_list):
    """重采样30帧"""
    ori = len(feat_list)
    if ori == 0:
        return [np.zeros(INPUT_DIM)] * SEQ_LEN
    idx = np.linspace(0, ori-1, SEQ_LEN, dtype=int)
    return [feat_list[i] for i in idx]

def norm_skeleton(seq):
    """髋部中心+肩宽归一化"""
    seq_reshape = seq.reshape(SEQ_LEN, 33, 4)
    left_hip = seq_reshape[:, 23, :2]
    right_hip = seq_reshape[:, 24, :2]
    hip_center = (left_hip + right_hip) / 2.0
    left_sho = seq_reshape[:, 11, :2]
    right_sho = seq_reshape[:, 12, :2]
    sho_width = np.linalg.norm(left_sho - right_sho, axis=-1, keepdims=True)
    sho_width[sho_width < 1e-6] = 1e-6
    seq_reshape[..., :2] = (seq_reshape[..., :2] - hip_center[:, None, :]) / sho_width[:, None, :]
    return seq_reshape.reshape(SEQ_LEN, INPUT_DIM)

def video_to_input(video_path):
    """输入视频 -> [30,132] 标准化骨架序列"""
    cap = cv2.VideoCapture(video_path)
    feat_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        feat_list.append(frame_to_vec(frame))
    cap.release()
    resampled = resample_seq(feat_list)
    arr = np.array(resampled, dtype=np.float32)
    norm_arr = norm_skeleton(arr)
    return arr

# ===================== 4. 批量测试集评估函数（必做项） =====================
def evaluate_test_set(model):
    print("="*60)
    print("【测试集批量评估】")
    test_ds = BadmintonSkeletonDataset(
        os.path.join(DATA_DIR, "14_X_test.npy"),
        os.path.join(DATA_DIR, "14_y_test.npy")
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    all_pred = []
    all_true = []

    model.eval()
    with torch.no_grad():
        for seq, label in test_loader:
            seq = seq.to(DEVICE)
            logits = model(seq)
            pred = torch.argmax(logits, dim=1)
            all_pred.extend(pred.cpu().numpy())
            all_true.extend(label.cpu().numpy())

    # 1. 总体准确率
    acc = accuracy_score(all_true, all_pred)
    print(f"测试集总体准确率 Accuracy: {acc:.4f}")
    print("-"*40)

    # 2. 分类报告
    print("【分类报告 Classification Report】")
    print(classification_report(all_true, all_pred, target_names=CLASS_NAMES))

    # 3. 混淆矩阵 + 绘图
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predict")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("./14_confusion_matrix.png")
    plt.show()
    print("✅ 混淆矩阵图已保存为 14_confusion_matrix.png")
    print("="*60)

# ===================== 5. 单视频推理函数（必做项） =====================
def infer_single_video(model, video_path):
    print("\n【单视频推理】")
    skeleton_seq = video_to_input(video_path)
    input_tensor = torch.from_numpy(skeleton_seq).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(prob, dim=1)

    pred_class = ID_TO_NAME[int(pred_idx.item())]
    confidence = float(conf.item())
    print(f"视频路径: {video_path}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.2f}")

# ===================== 【可选项】骨架可视化 + 生成演示视频 =====================
def generate_visual_demo_video(model, input_video_path, out_video_path="./14_visual_demo.mp4"):
    """
    逐帧绘制骨架+叠加预测标签，输出演示视频（加分项）
    """
    full_skeleton = video_to_input(input_video_path)
    input_tensor = torch.from_numpy(full_skeleton).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        prob = torch.softmax(logits, dim=1)
        conf_score, pred_index = torch.max(prob, dim=1)

    pred_name = ID_TO_NAME[int(pred_index.item())]
    show_text = f"Action: {pred_name} | Conf: {conf_score.item():.2f}"

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (frame_w, frame_h))

    print(f"\n【可选项-可视化演示视频生成中...】")
    print(f"全局预测结果: {show_text}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=pose_results.pose_landmarks,
                connections=mp_pose_module.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

        cv2.putText(
            img=frame,
            text=show_text,
            org=(20, 45),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 255, 0),
            thickness=2
        )
        video_writer.write(frame)

    cap.release()
    video_writer.release()
    print(f"✅ 可视化演示视频已保存至: {out_video_path}")

# ===================== 主入口 =====================
if __name__ == "__main__":
    # 1. 加载模型
    model = SkeletonTransformer(
        INPUT_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, NUM_CLASSES, DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"成功加载模型权重: {MODEL_PATH}")

    # 2. 测试集评估（必做）
    evaluate_test_set(model)

    # 3. 单视频推理（必做）
    infer_single_video(model, TEST_VIDEO_PATH)

    # ========== 可视化视频生成 ==========
    generate_visual_demo_video(model, TEST_VIDEO_PATH, out_video_path="./14_visual_demo.mp4")

    # 释放资源
    pose.close()
