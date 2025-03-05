# 手势控制音量系统

这是一个基于计算机视觉的手势控制音量系统，使用 MediaPipe 进行手势识别和人脸检测，通过手势来控制系统音量。

## 功能特点

- 实时手势识别
- 人脸检测和遮罩
- 音量平滑控制
- 可视化音量显示
- 实时 FPS 显示

## 环境要求

- Python 3.7+
- OpenCV
- MediaPipe
- pycaw
- numpy

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd hand_control
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：
```bash
python hand_control_volume.py
```

## 控制说明

- 使用大拇指和食指的距离来控制音量
- 距离越大，音量越大
- 按 ESC 键退出程序

## 参数调整

可以在 `HandControlVolume` 类的 `__init__` 方法中调整以下参数：

- `face_mask_color`: 人脸遮罩颜色
- `face_mask_alpha`: 人脸遮罩透明度
- `gesture_threshold`: 手势检测阈值
- `smooth_factor`: 音量平滑因子

## 许可证

MIT License 