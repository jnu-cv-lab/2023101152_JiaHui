# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./14_data"

# 模型超参数
INPUT_DIM = 132
SEQ_LEN = 30
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256
NUM_CLASSES = 6
DROPOUT = 0.1

# 训练超参数
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 20

# ===================== 1. 自定义数据集 =====================
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

# ===================== 2. 修复版位置编码（彻底解决维度问题） =====================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 生成位置索引: [max_len]
        position = torch.arange(0, max_len, dtype=torch.float)
        # 计算衰减项: [d_model//2]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # 初始化位置编码: [1, max_len, d_model] 适配 batch_first
        pe = torch.zeros(1, max_len, d_model)
        # 维度对齐运算
        pe[0, :, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[0, :, 1::2] = torch.cos(position.unsqueeze(1) * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

# ===================== 3. Skeleton Transformer 模型 =====================
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
        # 输入 [B, 30, 132]
        x = self.linear_embedding(x)    # [B, 30, 128]
        x = self.pos_encoder(x)        # 加入位置编码
        x = self.transformer_encoder(x)# Transformer 编码
        x = torch.mean(x, dim=1)       # 全局平均池化 [B, 128]
        x = self.dropout(x)
        out = self.classifier(x)       # [B, 6]
        return out

# ===================== 4. 训练主流程 =====================
def main():
    # 加载数据集
    train_dataset = BadmintonSkeletonDataset(
        os.path.join(DATA_DIR, "14_X_train.npy"),
        os.path.join(DATA_DIR, "14_y_train.npy")
    )
    test_dataset = BadmintonSkeletonDataset(
        os.path.join(DATA_DIR, "14_X_test.npy"),
        os.path.join(DATA_DIR, "14_y_test.npy")
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 模型、损失、优化器
    model = SkeletonTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 记录指标
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    print(f"训练设备: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for batch_seq, batch_label in pbar:
            batch_seq = batch_seq.to(DEVICE)
            batch_label = batch_label.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_seq)
            loss = criterion(logits, batch_label)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * batch_seq.size(0)
            pred = torch.argmax(logits, dim=1)
            train_correct += (pred == batch_label).sum().item()
            train_total += batch_seq.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / train_total
        avg_train_acc = train_correct / train_total

        # 测试集评估
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_seq, batch_label in test_loader:
                batch_seq = batch_seq.to(DEVICE)
                batch_label = batch_label.to(DEVICE)
                logits = model(batch_seq)
                pred = torch.argmax(logits, dim=1)
                test_correct += (pred == batch_label).sum().item()
                test_total += batch_seq.size(0)

        avg_test_acc = test_correct / test_total

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        test_acc_list.append(avg_test_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.4f} | "
              f"Test Acc: {avg_test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "./14_skeleton_transformer.pth")
    print("\n✅ 模型已保存为: 14_skeleton_transformer.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label="Train Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list, label="Train Acc", color="blue")
    plt.plot(test_acc_list, label="Test Acc", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./14_train_curve.png")
    plt.show()
    print("✅ 训练曲线已保存为: 14_train_curve.png")

if __name__ == "__main__":
    main()
