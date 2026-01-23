#!/usr/bin/env python
"""
AICL应用主入口脚本
用于运行完整的视频分析流水线
"""

import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_thumos import Config  # 导入现有的配置
from application.config import AppConfig, load_config_from_file
from application.main_pipeline import run_application


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AICL Video Analysis Application')
    
    parser.add_argument(
        '--video_path', 
        type=str, 
        required=True, 
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None, 
        help='Path to trained AICL model weights (.pth file). If not provided, will try to load from output directory.'
    )
    
    parser.add_argument(
        '--output_path', 
        type=str, 
        default=None, 
        help='Path for output annotated video (optional)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5, 
        help='Actionness threshold for temporal proposal extraction (default: 0.5)'
    )
    
    parser.add_argument(
        '--trigger_threshold', 
        type=float, 
        default=0.6, 
        help='Threshold for temporal trigger (default: 0.6)'
    )
    
    parser.add_argument(
        '--min_duration', 
        type=float, 
        default=16, 
        help='Minimum duration for temporal proposals in frames (default: 16)'
    )
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        default=None, 
        help='Path to configuration file (optional)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda', 
        choices=['cuda', 'cpu'], 
        help='Device to run inference on (default: cuda)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    print("=" * 60)
    print("AICL Video Analysis Application")
    print("Input video -> Feature Extraction -> AICL/Actionness Model -> ")
    print("High-confidence Segments -> YOLO Human Detection -> Pose Estimation -> ")
    print("Action/Risk/Violation Classification")
    print("=" * 60)
    
    # 解析参数
    args = parse_arguments()
    
    # 设置设备环境变量
    os.environ['CUDA_AVAILABLE'] = 'true' if args.device == 'cuda' else 'false'
    
    # 加载配置
    if args.config_path:
        config = load_config_from_file(args.config_path)
    else:
        # 使用默认配置
        config = AppConfig()
        config.actionness_threshold = args.threshold
        config.trigger_threshold = args.trigger_threshold
        config.min_duration = args.min_duration
    
    # 如果没有提供模型路径，尝试从output目录加载
    model_path = args.model_path
    if model_path is None:
        # 尝试从output目录查找模型
        possible_paths = [
            'output/CAS_Only.pkl',
            'output/model_rgb.pth',
            'output/model_flow.pth'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                print(f"Using model from: {model_path}")
                break
    
    # 验证输入文件
    if not os.path.exists(args.video_path):
        print(f"Error: Video file does not exist: {args.video_path}")
        sys.exit(1)
    
    if model_path and not os.path.exists(model_path):
        print(f"Error: Model file does not exist: {model_path}")
        sys.exit(1)
    
    print(f"Input video: {args.video_path}")
    if model_path:
        print(f"AICL model: {model_path}")
    else:
        print("AICL model: Using default (will attempt to load from output directory)")
    print(f"Actionness threshold: {args.threshold}")
    print(f"Device: {args.device}")
    
    try:
        # 运行应用程序
        results, output_video = run_application(
            video_path=args.video_path,
            model_path=model_path,
            config=config,
            output_video=args.output_path,
            actionness_threshold=args.threshold
        )
        
        print(f"\nProcessing completed successfully!")
        print(f"Output video saved to: {output_video}")
        
        # 保存详细结果
        import json
        results_summary = {
            'input_video': args.video_path,
            'output_video': output_video,
            'temporal_proposals_count': len(results['temporal_proposals']),
            'frames_with_detections': len(results['yolo_detections']),
            'total_poses_detected': sum(len(poses) for poses in results['pose_estimations'].values()),
            'detected_actions': {}
        }
        
        # 统计检测到的动作
        for frame_classifications in results['classifications'].values():
            for cls_info in frame_classifications:
                action = cls_info['action']
                if action not in results_summary['detected_actions']:
                    results_summary['detected_actions'][action] = 0
                results_summary['detected_actions'][action] += 1
        
        # 保存结果摘要
        summary_path = output_video.replace('.mp4', '_summary.json') if output_video.endswith('.mp4') else f"{output_video}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"Results summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()