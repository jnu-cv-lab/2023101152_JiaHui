#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    // 测试图片路径
    std::string imagePath = "/mnt/c/Users/Lenovo/Pictures/Camera Roll/test.shoutao.jpg";
    // 任务1：使用OpenCV读取一张测试图片
    cv::Mat image = cv::imread(imagePath);
    
    // 检查图片是否成功读取
    if (image.empty()) {
        std::cout << "错误：无法读取图片！请检查路径：" << imagePath << std::endl;
        std::cout << "提示：请确保文件存在且路径正确。" << std::endl;
        return -1;
    }
    
    // 任务2：输出图像基本信息
    std::cout << "========== 图像基本信息 ==========" << std::endl;
    std::cout << "图像宽度 (Width): " << image.cols << " 像素" << std::endl;
    std::cout << "图像高度 (Height): " << image.rows << " 像素" << std::endl;
    std::cout << "图像通道数 (Channels): " << image.channels() << std::endl;
    std::cout << "图像数据类型: ";
    
    // 输出数据类型
    switch (image.type()) {
        case CV_8UC1: std::cout << "CV_8UC1 (8位无符号单通道)"; break;
        case CV_8UC3: std::cout << "CV_8UC3 (8位无符号三通道)"; break;
        case CV_8UC4: std::cout << "CV_8UC4 (8位无符号四通道)"; break;
        case CV_16UC1: std::cout << "CV_16UC1 (16位无符号单通道)"; break;
        case CV_16UC3: std::cout << "CV_16UC3 (16位无符号三通道)"; break;
        case CV_32FC1: std::cout << "CV_32FC1 (32位浮点单通道)"; break;
        case CV_32FC3: std::cout << "CV_32FC3 (32位浮点三通道)"; break;
        default: std::cout << "其他类型 (type: " << image.type() << ")"; break;
    }
    std::cout << std::endl;
    std::cout << "总像素数: " << image.total() << std::endl;
    std::cout << "图像大小: " << image.total() * image.elemSize() << " 字节" << std::endl;
    std::cout << "===================================" << std::endl;
    
    // 任务3：显示原图
    cv::namedWindow("原图", cv::WINDOW_NORMAL);
    cv::imshow("原图", image);
    std::cout << "正在显示原图，按任意键继续..." << std::endl;
    cv::waitKey(0);
    
    // 任务4：转换为灰度图
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        std::cout << "已将彩色图转换为灰度图" << std::endl;
    } else if (image.channels() == 1) {
        grayImage = image.clone();
        std::cout << "图片已是灰度图" << std::endl;
    } else {
        std::cout << "图片通道数不是1或3，无法直接转换为灰度图" << std::endl;
        grayImage = image.clone();
    }
    
    // 显示灰度图
    cv::namedWindow("灰度图", cv::WINDOW_NORMAL);
    cv::imshow("灰度图", grayImage);
    std::cout << "正在显示灰度图，按任意键继续..." << std::endl;
    cv::waitKey(0);
    
    // 任务5：保存处理结果
    std::string savePath = "C:\\Users\\Lenovo\\Pictures\\Camera Roll\\test.shoutao_gray.jpg";
    bool success = cv::imwrite(savePath, grayImage);
    if (success) {
        std::cout << "灰度图已保存至: " << savePath << std::endl;
    } else {
        // 如果原路径保存失败，尝试保存到当前目录
        savePath = "gray_image.jpg";
        success = cv::imwrite(savePath, grayImage);
        if (success) {
            std::cout << "灰度图已保存至当前目录: " << savePath << std::endl;
        } else {
            std::cout << "保存失败！请检查目录权限。" << std::endl;
        }
    }
    
    // 任务6：使用NumPy做一个简单操作
    std::cout << "\n========== NumPy风格操作 ==========" << std::endl;
    
    // 操作1：输出某个像素值
    int centerX = image.cols / 2;
    int centerY = image.rows / 2;
    if (image.channels() == 3) {
        cv::Vec3b pixel = image.at<cv::Vec3b>(centerY, centerX);
        std::cout << "中心点 (" << centerX << ", " << centerY << ") 的像素值 (BGR): "
                  << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << std::endl;
    } else if (image.channels() == 1) {
        uchar pixel = image.at<uchar>(centerY, centerX);
        std::cout << "中心点 (" << centerX << ", " << centerY << ") 的像素值: " 
                  << (int)pixel << std::endl;
    }
    
    // 操作2：裁剪图像左上角一块区域（300x300像素）
    int cropWidth = std::min(300, image.cols);
    int cropHeight = std::min(300, image.rows);
    cv::Rect cropRegion(0, 0, cropWidth, cropHeight);
    cv::Mat croppedImage = image(cropRegion);
    
    // 显示裁剪区域
    cv::namedWindow("裁剪区域 (左上角)", cv::WINDOW_NORMAL);
    cv::imshow("裁剪区域 (左上角)", croppedImage);
    
    // 保存裁剪区域
    std::string cropSavePath = "cropped_region.jpg";
    cv::imwrite(cropSavePath, croppedImage);
    std::cout << "裁剪区域 (" << cropWidth << "x" << cropHeight << " 像素) 已保存至: " 
              << cropSavePath << std::endl;
    
    // 操作3：图像像素值统计（类似NumPy的统计操作）
    cv::Scalar mean, stddev;
    cv::meanStdDev(grayImage, mean, stddev);
    std::cout << "灰度图统计信息:" << std::endl;
    std::cout << "  均值: " << mean[0] << std::endl;
    std::cout << "  标准差: " << stddev[0] << std::endl;
    
    std::cout << "===================================" << std::endl;
    
    // 显示裁剪区域
    std::cout << "\n正在显示裁剪区域，按任意键退出..." << std::endl;
    cv::waitKey(0);
    
    // 关闭所有窗口
    cv::destroyAllWindows();
    
    std::cout << "\n程序执行完成！" << std::endl;
    
    return 0;
}
