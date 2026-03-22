# 2023101152_JiaHui
used to submit computer vision jobs
# 计算机视觉作业仓库
**学号**：2023101152  
**姓名**：莫嘉慧  
**课程**：计算机视觉

## 目录
| 编号 | 作业名称                | 文件路径          |
|----------|-------------------------|---------------------|
| 作业1    | YCbCr下采样与插值实验   | ./zuoye1/zuoye1.py     | 
| 实验报告1 | Python视觉开发环境搭建与图像基本读写 | ./lab report/report1.py | 


###在对应的文件夹中有更详细的说明，以下仅作简单介绍


## 作业1：YCbCr下采样与插值实验
### 实验目的
1. 掌握 OpenCV 实现图像色彩空间转换的方法
2. 理解 YCbCr 色彩空间亮度/色度分离的原理
3. 实现色度通道下采样与插值恢复，并通过 PSNR 量化图像质量

### 运行环境
- Python 3.7+
- 依赖库：`opencv-python`、`numpy`
- 安装命令：`pip install opencv-python numpy`

### 实验说明
#### 输入输出文件
- 原始图像路径：`C:\Users\Lenovo\Pictures\Camera Roll\test.picture.jpg`
- 处理后图像路径：`C:\Users\Lenovo\Pictures\Camera Roll\reconstructed.jpg`

#### 核心步骤
1. 读取 BGR 格式原始图像，验证路径有效性
2. 转换为 YCrCb 色彩空间，拆分 Y（亮度）、Cr/Cb（色度）通道
3. 对 Cr/Cb 通道进行 2 倍下采样（行列间隔取值）
4. 采用双线性插值（INTER_LINEAR）将色度通道恢复至原尺寸
5. 合并通道并转回 BGR 格式，生成重建图像
6. 计算原图与重建图的 PSNR 值，评估图像质量
7. 保存重建图像到指定路径

#### 实验结果
- 原始图像尺寸：4800*3200
- 下采样比例：4:2:0（Cr/Cb 通道分辨率为 Y 通道的 1/4）
- 插值方法：双线性插值（INTER_LINEAR）
- PSNR 值：52.29 dB


## 实验1

### 项目简介
本仓库包含计算机视觉课程的系列实验，从基础图像操作到进阶视觉算法实现，适配 WSL (Ubuntu) 环境，基于 Python + OpenCV 构建。
### 实验列表
### Lab 01：图像基本操作
#### 实验目标
- 掌握 OpenCV 图像读取、显示、保存核心方法
- 实现彩色图转灰度图、像素读取、图像裁剪等基础操作
- 适配 WSL 访问 Windows 本地文件路径

#### 快速运行
1. 激活虚拟环境：
   ```bash
   source .venv-basic/bin/activate
   ```
2. 进入实验目录：
   ```bash
   cd lab01-image-basic
   ```
3. 运行核心代码：
   ```bash
   python image_operation.py
   ```
#### 核心依赖
```bash
pip install opencv-python matplotlib numpy
```
#### 关键说明
- 测试图片路径：`/mnt/c/Users/Lenovo/Pictures/Camera Roll/test.cat.jpg`（WSL 格式）
- 生成文件：`grayscale_result.jpg`（灰度图）、`cropped_result.jpg`（裁剪图）
- 适配 WSL 环境，解决图像显示/路径访问核心问题
- 
### 环境要求
- Python 3.8+
- WSL (Ubuntu) 或 Linux 环境
- 虚拟环境：`.venv-basic`（推荐）
- 
### 目录结构
```
cv-course/
├── .venv-basic/          # 项目虚拟环境
├── lab01-image-basic/    # 图像基本操作实验
│   ├── image_operation.py # 实验核心代码
│   └── README.md         # 实验详细说明
└── README.md             # 仓库总览
```
### 注意事项
1. 运行前确保激活虚拟环境，避免依赖冲突
2. 图片路径需根据本地实际路径修改
3. WSL 环境下图像显示窗口可能无法弹出，可直接查看生成的图片文件
