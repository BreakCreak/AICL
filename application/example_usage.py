"""
AICL应用示例脚本
展示如何使用完整的视频分析流水线
"""

import os
import sys
import torch
import numpy as np
import cv2

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from config.config_thumos import Config
import argparse
from application.main_pipeline import AICLPipeline


def create_example_config():
    """创建示例配置"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--modal', type=str, default='all')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--exp_name', type=str, default='example_run')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--class_th', type=float, default=0.25)
    parser.add_argument('--q_val', type=float, default=0.7)
    parser.add_argument('--scale', type=int, default=24)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--feature_fps', type=int, default=25)
    parser.add_argument('--num_segments', type=int, default=750)
    parser.add_argument('--num_segments1', type=int, default=50)
    parser.add_argument('--num_segments2', type=int, default=1500)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--inference_only', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='AICL')
    parser.add_argument('--detection_inf_step', type=int, default=50)
    parser.add_argument('--soft_nms', action='store_true', default=False)
    parser.add_argument('--nms_alpha', type=float, default=0.35)
    parser.add_argument('--nms_thresh', type=float, default=0.6)
    parser.add_argument('--load_weight', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args([])
    return Config(args)


def main():
    print("="*60)
    print("AICL Video Analysis Application - Example Usage")
    print("="*60)
    
    # 创建配置
    config = create_example_config()
    
    print("Available modules in the AICL pipeline:")
    print("1. Video Processor - Handles video preprocessing and temporal proposal extraction")
    print("2. AICL Model - Performs actionness prediction using trained weights")
    print("3. YOLO Detector - Detects humans in high-confidence segments")
    print("4. Pose Estimator - Estimates human poses/keypoints")
    print("5. Action Classifier - Classifies actions/risk/violations")
    
    print("\nExample usage scenarios:")
    print("- Real-time video surveillance for anomaly detection")
    print("- Sports activity analysis")
    print("- Healthcare monitoring (fall detection)")
    print("- Security applications")
    
    print("\nTo run the full pipeline, use:")
    print("python application/run_app.py --video_path VIDEO_FILE --model_path MODEL_WEIGHTS")
    
    print("\nFor programmatic usage:")
    print("# Create pipeline instance")
    print("# pipeline = AICLPipeline(model_path='path/to/weights', config=config)")
    print("# results = pipeline.run_inference('path/to/video.mp4')")
    
    # Show a brief demo of pipeline creation (without running inference)
    print("\n" + "="*60)
    print("Pipeline Structure Demo (without actual inference):")
    print("="*60)
    
    try:
        # 创建管道实例（不加载实际权重）
        print("Creating AICL Pipeline instance...")
        pipeline = AICLPipeline(
            model_path=None,  # 不加载权重
            config=config,
            yolo_model='yolov8n.pt',
            pose_model='yolov8n-pose.pt'
        )
        print("✓ Pipeline created successfully")
        
        print("\nPipeline components:")
        print(f"- Video Processor: {type(pipeline.video_processor).__name__}")
        print(f"- AICL Model: {type(pipeline.aicl_model).__name__}")
        print(f"- YOLO Detector: {type(pipeline.yolo_detector).__name__}")
        print(f"- Pose Estimator: {type(pipeline.pose_estimator).__name__}")
        print(f"- Action Classifier: {type(pipeline.action_classifier).__name__}")
        
        print("\nThe pipeline is ready to process videos with the following workflow:")
        print("1. Extract video features")
        print("2. Run AICL model inference")
        print("3. Extract temporal proposals from high-actionness segments")
        print("4. Run YOLO detection on temporal proposals")
        print("5. Estimate poses in detected human regions")
        print("6. Classify actions/risk/violations")
        
    except Exception as e:
        print(f"Demo pipeline creation failed: {e}")
        print("This is expected if models are not downloaded yet.")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("Refer to application/README.md for detailed usage instructions.")
    print("="*60)


if __name__ == "__main__":
    main()