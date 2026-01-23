# AICL 视频分析应用

这是一个完整的视频分析流水线，实现了从输入视频到动作/违规/风险判定的全过程。

## 架构概述

整体架构（高层逻辑）：
1. **输入视频** → 视频文件输入
2. **特征提取**（I3D / SlowFast / ViT 等）→ 提取视频时空特征
3. **AICL / Actionness 时序模型** → 生成动作置信度
4. **【动作置信度 + 时间段】** → 生成高置信度时间片段
5. **（仅对高置信片段）YOLO 人体检测** → 检测人体位置
6. **人体关键点检测**（OpenPose / HRNet / RTMPose）→ 提取人体姿态
7. **动作级 / 违规级 / 风险级判定** → 最终分类

## 主要特性

### 1. 动作触发器（Trigger）
- 当 actionness > θ 且连续 T 帧激活下游检测
- 示例：`if mean(actionness[t:t+16]) > 0.6: trigger = True`

### 2. 时间片段裁剪（Temporal Proposal）
- 合并连续高 actionness 区间
- 得到 [t_start, t_end] 时间段
- 示例：视频 0–3000 帧 AICL 输出：[320–480], [960–1080], [1820–1950]
- 只对这些区间跑 YOLO + Keypoints

### 3. YOLO人体检测的作用
- **确认“是否真的有人”**：避免误触发（比如风吹树、背景运动）
  - AICL 说：这里像动作
  - YOLO 说：这里没有人 → 丢弃
- **人框裁剪给关键点用**：提高精度和速度

## 文件结构

```
application/
├── __init__.py
├── main_pipeline.py      # 主流水线
├── video_processor.py    # 视频处理器
├── aicl_model.py         # AICL模型接口
├── yolo_detector.py      # YOLO人体检测器
├── pose_estimator.py     # 姿态估计器
├── action_classifier.py  # 动作分类器
├── config.py             # 应用配置
├── run_app.py           # 主入口脚本
├── test_app.py          # 测试脚本
├── example_usage.py     # 示例用法
├── README.md            # 本说明文档
└── requirements_app.txt # 依赖包
```

## 使用方法

### 安装依赖
```bash
pip install -r requirements_app.txt
```

### 运行应用
```bash
python application/run_app.py --video_path PATH_TO_VIDEO --model_path PATH_TO_TRAINED_WEIGHTS
```

### 自动使用output目录中的模型
如果您将训练好的模型放在output目录中，应用程序会自动检测并使用它们：
- AICL模型: `output/CAS_Only.pkl`, `output/model_rgb.pth`, 或 `output/model_flow.pth`
- YOLO模型: `output/BeltDetection(yolo11n).pt` 或 `output/yolov8n-pose.pt`
- 姿态模型: `output/yolov8n-pose.pt`

### 参数说明
- `--video_path`: 输入视频路径（必需）
- `--model_path`: 训练好的AICL模型权重路径（可选，如果不提供会自动从output目录查找）
- `--output_path`: 输出视频路径（可选）
- `--threshold`: Actionness阈值（默认0.5）
- `--trigger_threshold`: 触发阈值（默认0.6）
- `--min_duration`: 最小持续时间（默认16帧）
- `--device`: 设备（cuda/cpu，默认cuda）

### 示例
```bash
# 使用指定模型
python application/run_app.py \
    --video_path ./sample_video.mp4 \
    --model_path ./output/CAS_Only.pkl \
    --output_path ./output/annotated_video.mp4 \
    --threshold 0.5 \
    --device cuda

# 自动从output目录加载模型
python application/run_app.py \
    --video_path ./sample_video.mp4 \
    --output_path ./output/annotated_video.mp4 \
    --threshold 0.5 \
    --device cuda
```

## 开发说明

### 模块功能
- `VideoProcessor`: 视频预处理、特征提取、时间片段提取
- `AICLInference`: AICL模型推理接口
- `YOLODetector`: 人体检测
- `PoseEstimator`: 姿态估计及特征分析
- `ActionClassifier`: 动作/违规/风险分类

### 下游模型执行条件
仅在YOLO人体检测确认存在真实人体的情况下，才执行后续的关键点检测和动作判定。若AICL预测有动作但YOLO未检测到人体，则丢弃该提案以避免误触发。

### GPU加速
- 所有模型默认使用GPU加速（如果可用）
- 通过`device`参数控制

## 技术栈
- PyTorch: 深度学习框架
- OpenCV: 图像处理
- Ultralytics YOLO: 人体检测和姿态估计
- Scikit-learn: 分类器
- NumPy, SciPy: 数值计算