# 2023101152_JiaHui
used to submit computer vision jobs
# 计算机视觉作业仓库
**学号**：2023101152  
**姓名**：莫嘉慧  
**课程**：计算机视觉

## 目录
| 编号 | 作业名称                | 文件路径          |
|----------|-------------------------|---------------------|
| 作业2    | YCbCr下采样与插值实验（第二周理论课）   | ./zuoye1/zuoye1.py     | 
| 作业3    | 图像增强处理（第三周理论课）   | ./zuoye2/zuoye2.py     | 
| 作业4    | DFT&DCT与图像处理关系（第四周理论课）   | ./zuoye 4/zuoye4.py     | 
| 作业5    |图像局部频率FFT法与空域梯度法对比（第五周理论课）   | ./zuoye5/zuoye5.py     | 

| 实验报告2 | Python视觉开发环境搭建与图像基本读写（第二周实验课） | ./lab report/report1.py | 
| 实验报告3 |OpenCV环境搭建与c++图像基本读写（第三周实验课） | ./lab report2/main.cpp | 
| 实验报告4 |图像下采样与恢复（第四周实验课） | ./lab week4/report.py | 
| 实验报告5 |下采样抗混叠与自适应滤波实验（第五周实验课） | ./lab week5/lab5.py | 


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


## 作业2：图像增强处理
基于 Python + OpenCV 实现经典图像增强算法，对**原图、低对比度、高斯噪声、椒盐噪声**四类图像进行对比处理，包含：
- 自编全局直方图均衡化（HE）
- CLAHE 自适应均衡化
- 均值/高斯/中值滤波
- Laplacian 锐化
- 组合增强：Filter→HE / HE→Filter

## 评价指标
- **信息熵**：衡量图像信息丰富度与对比度
- **平均梯度**：衡量图像清晰度（噪声会表现为异常高梯度）

## 实验结论
- 低对比度图像：**CLAHE 效果最优**
- 高斯噪声：**高斯滤波去噪最佳**
- 椒盐噪声：**中值滤波效果显著最优**
- 锐化仅适用于清晰无噪图像
- 复合退化推荐：**先滤波再均衡（Filter→HE）**

## 运行
```bash
pip install opencv-python numpy matplotlib
python 实验代码.py
```
生成 `result_all.png` 对比图及终端定量指标。


## 实验1：图像基本读写

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
# cv-course
计算机视觉 / 数字图像处理基础实验库

以下是简洁版 README.md，适合放在作业文件夹中：

# 实验二

基于 OpenCV 的 C++ 图像处理程序，实现图片读取、灰度转换、裁剪保存等基本操作。

## 文件说明

- `main.cpp` - 主程序源代码
- `test.shoutao.jpg` - 原始测试图片
- `gray_image.jpg` - 处理后生成的灰度图
- `cropped_region.jpg` - 裁剪的左上角区域图片
- `readme.md` - 本说明文件

## 功能

1. 读取测试图片
2. 输出图像尺寸、通道数、数据类型等信息
3. 显示原图、灰度图、裁剪区域
4. 彩色图转灰度图
5. 保存灰度图和裁剪区域
6. 输出中心像素值及图像统计信息

## 编译运行

```bash
# 编译
g++ -g main.cpp -o main -I/usr/include/opencv4 \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

# 运行
./main
```

## 环境要求

- OpenCV 4.x
- GCC/G++ 编译器

## 输出

```
========== 图像基本信息 ==========
图像宽度 (Width): 960 像素
图像高度 (Height): 1440 像素
图像通道数 (Channels): 3
图像数据类型: CV_8UC3 (8位无符号三通道)
总像素数: 1382400
图像大小: 4147200 字节
===================================
正在显示原图，按任意键继续...
已将彩色图转换为灰度图
正在显示灰度图，按任意键继续...
灰度图已保存至: C:\Users\Lenovo\Pictures\Camera Roll\test.shoutao_gray.jpg

========== NumPy风格操作 ==========
中心点 (480, 720) 的像素值 (BGR): 144, 173, 178
裁剪区域 (300x300 像素) 已保存至: cropped_region.jpg
灰度图统计信息:
  均值: 124.215
  标准差: 57.1146
===================================
```
