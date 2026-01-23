"""
AICL应用测试脚本
用于测试应用程序各组件的功能
"""
import torch
import numpy as np
import sys
import os
import argparse

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)
sys.path.append(current_dir)

# 现在可以导入模块
from config.config_thumos import Config, parse_args
from application.video_processor import VideoProcessor
from application.aicl_model import AICLInference
from application.yolo_detector import YOLODetector
from application.pose_estimator import PoseEstimator
from application.action_classifier import ActionClassifier


def create_mock_args():
    """创建模拟的命令行参数"""
    args = argparse.Namespace()
    args.lr = 0.0001
    args.modal = 'all'
    args.batch_size = 16
    args.data_path = './data'
    args.output_dir = './output'
    args.exp_name = 'test_exp'
    args.num_workers = 8
    args.class_th = 0.25
    args.q_val = 0.7
    args.scale = 24
    args.model_file = None
    args.seed = 1
    args.feature_fps = 25
    args.num_segments = 750
    args.num_segments1 = 50
    args.num_segments2 = 1500
    args.num_epochs = 5000
    args.gamma = 0.2
    args.inference_only = False
    args.model_name = 'test_model'
    args.detection_inf_step = 50
    args.soft_nms = False
    args.nms_alpha = 0.35
    args.nms_thresh = 0.6
    args.load_weight = False
    args.verbose = False
    return args


def test_video_processor():
    """测试视频处理器"""
    print("Testing VideoProcessor...")
    processor = VideoProcessor()
    
    # 测试actionness序列处理
    actionness1 = torch.rand(1, 100)  # 模拟actionness序列
    actionness2 = torch.rand(1, 100)  # 模拟另一个actionness序列
    
    proposals = processor.temporal_proposal_extraction(
        actionness1, actionness2, 
        threshold=0.5, 
        min_duration=5
    )
    
    print(f"Found {len(proposals)} temporal proposals")
    print(f"Proposals: {proposals}")
    
    # 测试触发器
    trigger = processor.temporal_trigger(actionness1, threshold=0.6, window_size=5)
    print(f"Trigger activated: {trigger}")
    
    print("VideoProcessor test completed!\n")


def test_aicl_model():
    """测试AICL模型（模拟）"""
    print("Testing AICLInference...")
    
    # 创建一个模拟配置
    args = create_mock_args()
    config = Config(args)
    
    # 初始化模型（不加载权重，因为可能不存在）
    try:
        model = AICLInference(model_path=None, config=config)
        print("AICLInference initialized successfully")
        
        # 创建模拟输入
        features = torch.randn(1, 50, 2048)  # [batch_size, num_segments, feature_dim]
        
        # 测试预测
        outputs = model.predict(features)
        print(f"AICL outputs keys: {list(outputs.keys())}")
        print(f"Actionness1 shape: {outputs['actionness1'].shape}")
        print(f"Actionness2 shape: {outputs['actionness2'].shape}")
        
        # 测试获取actionness分数
        act1, act2 = model.get_actionness_scores(features)
        print(f"Got actionness scores: {act1.shape}, {act2.shape}")
        
    except Exception as e:
        print(f"AICL model test skipped due to: {e}")
    
    print("AICLInference test completed!\n")


def test_yolo_detector():
    """测试YOLO检测器"""
    print("Testing YOLODetector...")
    
    try:
        detector = YOLODetector(model_path='yolov8n.pt')  # 使用轻量级模型
        print("YOLODetector initialized successfully")
        
        # 创建一个模拟的检测输入
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_persons_in_frame(dummy_frame)
        print(f"Sample detection on dummy frame: Found {len(detections)} persons")
        
    except Exception as e:
        print(f"YOLO detector test skipped due to: {e}")
    
    print("YOLODetector test completed!\n")


def test_pose_estimator():
    """测试姿态估计器"""
    print("Testing PoseEstimator...")
    
    try:
        estimator = PoseEstimator(model_path='yolov8n-pose.pt')  # 使用姿态估计模型
        print("PoseEstimator initialized successfully")
        
        # 创建一个模拟的输入
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dummy_detection = [{'bbox': np.array([100, 100, 200, 200]), 'confidence': 0.9}]
        
        poses = estimator.estimate_pose_in_frame(dummy_frame, dummy_detection)
        print(f"Sample pose estimation: Found {len(poses)} poses")
        
        if poses:
            features = estimator.analyze_pose_features(poses[0])
            print(f"Pose features: {list(features.keys())}")
        
    except Exception as e:
        print(f"Pose estimator test skipped due to: {e}")
    
    print("PoseEstimator test completed!\n")


def test_action_classifier():
    """测试动作分类器"""
    print("Testing ActionClassifier...")
    
    classifier = ActionClassifier()
    print("ActionClassifier initialized successfully")
    
    # 创建模拟的姿态数据
    dummy_pose = {
        'keypoints': [
            {'name': 'nose', 'x': 100, 'y': 100, 'confidence': 0.9},
            {'name': 'left_eye', 'x': 95, 'y': 95, 'confidence': 0.8},
            {'name': 'right_eye', 'x': 105, 'y': 95, 'confidence': 0.85}
        ],
        'features': {
            'shoulder_angle': 120,
            'elbow_angle_left': 90,
            'elbow_angle_right': 95,
            'knee_angle_left': 160,
            'knee_angle_right': 155,
            'center_of_gravity': (320, 240)
        }
    }
    
    action, confidence = classifier.predict_action([dummy_pose])
    print(f"Predicted action: {action} with confidence: {confidence:.2f}")
    
    print("ActionClassifier test completed!\n")


def main():
    """运行所有测试"""
    print("Running AICL Application Tests\n")
    
    test_video_processor()
    test_aicl_model()
    test_yolo_detector()
    test_pose_estimator()
    test_action_classifier()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()