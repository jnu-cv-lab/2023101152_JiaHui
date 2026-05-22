# ===================== 基础环境 & 上次模型 =====================
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

plt.switch_backend('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理（完全和上次一致）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集（完全和上次一致）
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ===================== 任务1：复用上次 CNN 模型 =====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 训练函数（统一接口，方便优化器/学习率对比）
def train_model(opt_name, lr, epochs=5):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()

    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'SGD+Momentum':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    print(f"\n===== {opt_name} | lr={lr} =====")
    for epoch in range(epochs):
        model.train()
        t_loss, t_correct = 0.0, 0
        for img, lbl in train_loader:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            _, pred = torch.max(out, 1)
            t_correct += (pred == lbl).sum().item()

        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for img, lbl in val_loader:
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                v_loss += criterion(out, lbl).item()
                _, pred = torch.max(out, 1)
                v_correct += (pred == lbl).sum().item()

        tl = t_loss / len(train_loader)
        vl = v_loss / len(val_loader)
        ta = 100 * t_correct / len(train_subset)
        va = 100 * v_correct / len(val_subset)

        train_losses.append(tl)
        train_accs.append(ta)
        val_losses.append(vl)
        val_accs.append(va)
        print(f"Epoch {epoch+1} | TrainLoss:{tl:.3f} | TrainAcc:{ta:.2f}% | ValLoss:{vl:.3f} | ValAcc:{va:.2f}%")

    # 测试集
    test_correct = 0
    with torch.no_grad():
        for img, lbl in test_loader:
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            _, pred = torch.max(out, 1)
            test_correct += (pred == lbl).sum().item()
    test_acc = 100 * test_correct / len(test_dataset)
    print(f"✅ Test Acc: {test_acc:.2f}%")
    return model, train_losses, train_accs, val_losses, val_accs, test_acc

# ===================== 任务1：重新训练模型 =====================
print("="*60)
print("                任务1：复用模型并重新训练")
print("="*60)
model_best, tl_best, ta_best, vl_best, va_best, test_acc_best = train_model('Adam', 0.001, epochs=5)

# ===================== 任务2：优化器对比 SGD / SGD+M / Adam =====================
print("\n" + "="*60)
print("                任务2：优化器对比")
print("="*60)
optimizers = ['SGD', 'SGD+Momentum', 'Adam']
opt_results = {}
for opt in optimizers:
    _, tl, ta, vl, va, test_acc = train_model(opt, 0.01 if opt == 'SGD' else 0.001, epochs=5)
    opt_results[opt] = (tl, ta, vl, va, test_acc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for k,v in opt_results.items():
    plt.plot(v[0], label=f'{k} loss')
plt.title('Optimizer Train Loss')
plt.legend()

plt.subplot(1,2,2)
for k,v in opt_results.items():
    plt.plot(v[3], label=f'{k} val acc')
plt.title('Optimizer Val Acc')
plt.legend()
plt.savefig("11_optimizer_comparison.png")
plt.close()

# ===================== 任务3：学习率对比 Adam lr=0.1,0.01,0.001 =====================
print("\n" + "="*60)
print("                任务3：学习率对比")
print("="*60)
lrs = [0.1, 0.01, 0.001]
lr_results = {}
for lr in lrs:
    _, tl, ta, vl, va, test_acc = train_model('Adam', lr, epochs=5)
    lr_results[f'lr={lr}'] = (tl, ta, vl, va, test_acc)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
for k,v in lr_results.items():
    plt.plot(v[0], label=k)
plt.title('LR Train Loss')
plt.legend()

plt.subplot(1,2,2)
for k,v in lr_results.items():
    plt.plot(v[3], label=k)
plt.title('LR Val Acc')
plt.legend()
plt.savefig("11_lr_comparison.png")
plt.close()

# ===================== 任务4：卷积核可视化 =====================
print("\n" + "="*60)
print("                任务4：卷积核可视化")
print("="*60)
conv1_weights = model_best.conv1.weight.data.cpu().numpy()
plt.figure(figsize=(10,5))
for i in range(min(8, 16)):
    plt.subplot(2,4,i+1)
    plt.imshow(conv1_weights[i,0], cmap='gray')
    plt.title(f'Kernel {i+1}')
    plt.axis('off')
plt.savefig("11_conv1_kernels.png")
plt.close()

# ===================== 任务5：Feature map 可视化 =====================
print("\n" + "="*60)
print("                任务5：Feature Map 可视化")
print("="*60)
img, lbl = test_dataset[0]
input_img = img.unsqueeze(0).to(device)
with torch.no_grad():
    feat_map = model_best.conv1(input_img)
feat_map = feat_map.squeeze(0).cpu().numpy()

plt.figure(figsize=(10,5))
for i in range(min(8, 16)):
    plt.subplot(2,4,i+1)
    plt.imshow(feat_map[i], cmap='gray')
    plt.title(f'Feat {i+1}')
    plt.axis('off')
plt.savefig("11_feature_maps.png")
plt.close()

# ===================== 任务6：错误分类样本 =====================
print("\n" + "="*60)
print("                任务6：错误分类样本")
print("="*60)
error_images = []
error_labels = []
error_preds = []

model_best.eval()
with torch.no_grad():
    for img, lbl in test_loader:
        img = img.to(device)
        out = model_best(img)
        _, pred = torch.max(out, 1)
        for i in range(len(img)):
            if pred[i] != lbl[i]:
                error_images.append(img[i].cpu())
                error_labels.append(lbl[i].item())
                error_preds.append(pred[i].item())
                if len(error_images) >= 8:
                    break
        if len(error_images) >= 8:
            break

plt.figure(figsize=(12,6))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(error_images[i].squeeze(), cmap='gray')
    plt.title(f'True:{error_labels[i]}\nPred:{error_preds[i]}')
    plt.axis('off')
plt.savefig("11_error_samples.png")
plt.close()

# ===================== 任务7：混淆矩阵（纯matplotlib）=====================
print("\n" + "="*60)
print("                任务7：混淆矩阵")
print("="*60)
all_preds = []
all_labels = []
model_best.eval()
with torch.no_grad():
    for img, lbl in test_loader:
        img = img.to(device)
        out = model_best(img)
        _, pred = torch.max(out, 1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(lbl.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Pred')
plt.ylabel('True')
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.savefig("11_confusion_matrix.png")
plt.close()

print("\n🎉 任务1～7 全部完成！所有图表已以 11_ 开头保存！")
