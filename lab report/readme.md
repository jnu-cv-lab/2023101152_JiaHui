# 计算机视觉基础实验：图像基本操作
## 实验目的
1. 掌握 OpenCV 库读取、显示、保存图像的核心方法
2. 理解彩色图像转灰度图像的原理与实现
3. 熟悉 NumPy 对图像像素的基本操作（像素读取、图像裁剪）
4. 适配 WSL 环境访问 Windows 本地图片路径的方法

## 环境配置
### 前置条件
- 已安装 WSL (Ubuntu) 环境
- 已配置 Python 虚拟环境 `.venv-basic`
- 虚拟环境中已安装以下依赖库

### 依赖安装
在激活的虚拟环境中执行：
```bash
# 激活虚拟环境
source .venv-basic/bin/activate

# 安装依赖
pip install opencv-python matplotlib numpy
```

### 验证环境
创建 `check_env.py` 文件，执行以下代码验证依赖：
```python
import cv2
import numpy
import matplotlib

print(" OpenCV 版本:", cv2.__version__)
print(" NumPy 版本:", numpy.__version__)
print(" Matplotlib 版本:", matplotlib.__version__)
print(" 所有依赖库已安装完成！")
```

## 项目结构
```
cv-course/
├── .venv-basic/          # 虚拟环境目录
└── lab01-image-basic/    # 实验目录
    ├── image_operation.py # 核心实验代码
    ├── grayscale_result.jpg # 生成的灰度图
    ├── cropped_result.jpg   # 生成的裁剪图
    └── README.md            # 实验说明文档
```

## 运行步骤
1. **准备测试图片**
   将测试图片放在 Windows 路径：`C:\Users\Lenovo\Pictures\Camera Roll\test.cat.jpg`
   （WSL 中对应路径：`/mnt/c/Users/Lenovo/Pictures/Camera Roll/test.cat.jpg`）

2. **进入实验目录**
   ```bash
   cd cv-course/lab01-image-basic
   ```

3. **运行实验代码**
   ```bash
   # 确保虚拟环境已激活
   source ../.venv-basic/bin/activate

   # 执行代码
   python image_operation.py
   ```

## 代码功能说明
| 功能模块 | 核心代码 | 说明 |
|----------|----------|------|
| 图片读取 | `cv2.imread(image_path)` | 适配 WSL 访问 Windows 图片路径 |
| 图片信息输出 | `img.shape`/`img.dtype`/`img.size` | 输出图像尺寸、通道数、像素类型等 |
| 色彩空间转换 | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | 解决 OpenCV BGR 与 Matplotlib RGB 色差问题 |
| 灰度图转换 | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | 将彩色图转为单通道灰度图 |
| 图片保存 | `cv2.imwrite(save_path, img)` | 保存灰度图和裁剪图到实验目录 |
| 像素读取 | `img[y, x]` | 获取指定坐标的 BGR 像素值 |
| 图像裁剪 | `img[0:200, 0:200]` | 截取左上角 200×200 像素区域 |

## 实验结果
1. **终端输出**：显示图像基本信息、指定坐标像素值
2. **生成文件**：
   - `grayscale_result.jpg`：灰度转换后的图像
   - `cropped_result.jpg`：裁剪后的图像

## 常见问题解决
### 问题1：无法读取图片
- 原因：路径格式错误、图片不存在或权限问题
- 解决：
  1. 确认 WSL 路径格式：`/mnt/c/Users/...`（注意大小写和空格）
  2. 检查图片文件是否存在且命名正确
  3. 赋予图片读取权限：`chmod 644 /mnt/c/Users/Lenovo/Pictures/Camera\ Roll/test.cat.jpg`

### 问题2：图片显示颜色失真
- 原因：OpenCV 读取为 BGR 格式，Matplotlib 显示为 RGB 格式
- 解决：使用 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 转换后再显示

### 问题3：WSL 中无法弹出显示窗口
- 影响：不影响文件保存功能
- 替代方案：直接查看实验目录下生成的 `grayscale_result.jpg` 和 `cropped_result.jpg` 文件

## 核心知识点
1. OpenCV 读取图像默认采用 BGR 色彩空间，与主流显示工具的 RGB 格式不同，需手动转换
2. 图像在 OpenCV 中以 NumPy 数组形式存储，可通过数组操作实现像素级处理
3. WSL 访问 Windows 本地文件需使用 `/mnt/盘符/路径` 格式，空格需用 `\` 转义

## 注意事项
1. 运行代码前确保虚拟环境已激活
2. 测试图片路径需根据实际情况修改
3. 生成的结果文件保存在 `lab01-image-basic` 目录下
4. 若需修改裁剪区域大小，调整 `img[0:200, 0:200]` 中的数值即可

### 总结
1. 本实验完成了图像读取、信息输出、灰度转换、像素操作、裁剪保存等核心基础操作，覆盖了计算机视觉入门的核心知识点；
2. 重点适配了 WSL 环境访问 Windows 本地文件的路径规则，解决了跨环境操作的关键问题；
