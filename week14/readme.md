# README.md
```
├── 14_1.py                # 任务1：视频骨架预处理脚本
├── 14_2.py                # 任务2：Transformer模型训练脚本
├── 14_3.py                # 任务3：测试评估、单视频推理、可视化加分项
├── 14_X_train.npy         # 训练集骨架特征数据
├── 14_y_train.npy         # 训练集标签
├── 14_X_test.npy          # 测试集骨架特征数据
├── 14_y_test.npy          # 测试集标签
├── 14_label_map.json      # 数字标签与动作名称映射文件
├── 14_skeleton_transformer.pth  # 训练完成的模型权重
├── 14_train_curve.png     # 训练损失&准确率曲线图
├── 14_confusion_matrix.png # 测试集混淆矩阵热力图
└── 14_visual_demo.mp4     # 加分项：带骨架+预测标签的演示视频
```

### 简易运行顺序
1. `python 14_1.py` 生成数据集
2. `python 14_2.py` 训练模型
3. `python 14_3.py` 评估+推理
