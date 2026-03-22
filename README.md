# 2023101152_JiaHui
used to submit computer vision jobs
# 计算机视觉作业仓库
**学号**：2023101152  
**姓名**：莫嘉慧  
**课程**：计算机视觉

## 作业目录
| 作业编号 | 作业名称                | 文件路径               | 完成状态 |
|----------|-------------------------|------------------------|----------|
| 作业1    | YCbCr下采样与插值实验   | ./zuoye1/zuoye1.py     | ✅ 已完成 |


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
