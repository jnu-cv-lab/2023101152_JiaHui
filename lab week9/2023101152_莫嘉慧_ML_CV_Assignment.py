# ================== 任务1：数据准备 ==================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# 加载数据集
digits = load_digits()

# 查看图像数量
print("数据集中图像总数量：", digits.images.shape[0])
# 查看每张图像大小
print("每张图像的大小：", digits.images.shape[1], "×", digits.images.shape[2])
# 查看类别标签
print("数据集中的类别标签：", np.unique(digits.target))

# ----------------- 显示样本集图片 -----------------
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap="gray")
    plt.title(f"Label: {digits.target[i]}", fontsize=10) 
    plt.axis("off")
plt.suptitle("Handwritten Digits Samples", fontsize=14) 
plt.tight_layout()
plt.savefig("9_samples.png")  # 保存
plt.close()

# ================== 任务2：数据划分 ==================
# 特征向量和标签
X = digits.data
y = digits.target

# 划分训练集、测试集，测试集25%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("\n===== 数据集划分结果 =====")
print("训练集样本数量：", len(X_train))
print("测试集样本数量：", len(X_test))
print("测试集占比：{:.2f}%".format(len(X_test) / len(X) * 100))

print("\n===== 训练集与测试集作用 =====")
print("训练集：用于训练机器学习模型，让模型学习手写数字像素特征与标签的映射规律。")
print("测试集：不参与模型训练，用于模拟未知新样本，评估模型泛化分类性能。")

# ================== 任务3：特征表示 ==================
print("\n===== 任务3 特征表示 =====")
# 取单张8×8图像
sample_img = digits.images[0]
print("单张图像二维矩阵形状：", sample_img.shape)

# 展平转为一维特征向量
sample_vec = sample_img.flatten()
print("展平后特征向量维度：", sample_vec.shape)
print("单张图像前10维特征值：", sample_vec[:10])

# ================== 任务4：模型训练（6种分类器） ==================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("\n===== 任务4 模型训练与测试集准确率 =====")

# 1. KNN 最近邻
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"1. KNN 分类准确率：{acc_knn:.4f}")

# 2. 朴素贝叶斯
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
print(f"2. 朴素贝叶斯 分类准确率：{acc_nb:.4f}")

# 3. 逻辑回归
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"3. 逻辑回归 分类准确率：{acc_lr:.4f}")

# 4. SVM 支持向量机
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"4. SVM 分类准确率：{acc_svm:.4f}")

# 5. 决策树
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
print(f"5. 决策树 分类准确率：{acc_dt:.4f}")

# 6. 随机森林
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"6. 随机森林 分类准确率：{acc_rf:.4f}")

# ================== 任务6：优选模型错误分析（KNN） ==================
from sklearn.metrics import confusion_matrix

# 1. 绘制并保存混淆矩阵（纯matplotlib，英文标签）
cm = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 在混淆矩阵上标注数字
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("9_confusion_matrix.png")
plt.close()

# 2. 找出所有被错误分类的样本
error_idx = np.where(y_pred_knn != y_test)[0]
print("\n===== KNN模型错误分类样本分析 =====")
print(f"测试集错误样本总数：{len(error_idx)}")

# 3. 保存错误样本图片
plt.figure(figsize=(12,6))
for i, idx in enumerate(error_idx[:8]):
    plt.subplot(2,4,i+1)
    img = X_test[idx].reshape(8,8)
    plt.imshow(img, cmap="gray")
    plt.title(f"True:{y_test[idx]}\nPred:{y_pred_knn[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("9_error_samples.png")
plt.close()
