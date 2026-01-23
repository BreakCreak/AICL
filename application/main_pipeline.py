"""
AICL应用主流水线
整合视频处理、AICL模型、YOLO检测、姿态估计和动作分类的完整流程
"""
import os
import cv2
import torch
import numpy as np
from typing import List, Tuple, Dict
from .video_processor import VideoProcessor
from .aicl_model import AICLInference
from .yolo_detector import YOLODetector
from .pose_estimator import PoseEstimator
from .action_classifier import ActionClassifier


class AICLPipeline:
    """AICL完整应用流水线"""
    
    def __init__(self, 
                 model_path: str = None, 
                 config=None, 
                 yolo_model: str = None,
                 pose_model: str = None):
        """
        初始化AICL流水线
        :param model_path: AICL模型权重路径
        :param config: 配置对象
        :param yolo_model: YOLO模型路径
        :param pose_model: 姿态估计模型路径
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化各组件
        self.video_processor = VideoProcessor()
        self.aicl_model = AICLInference(model_path, config)
        self.yolo_detector = YOLODetector(yolo_model)
        self.pose_estimator = PoseEstimator(pose_model)
        self.action_classifier = ActionClassifier()
        
        self.config = config
    
    def run_inference(self, 
                     video_path: str, 
                     actionness_threshold: float = 0.5,
                     min_duration: int = 16,
                     trigger_threshold: float = 0.6,
                     trigger_window: int = 16) -> Dict:
        """
        运行完整的推理流程
        :param video_path: 视频路径
        :param actionness_threshold: actionness阈值
        :param min_duration: 最小持续时间
        :param trigger_threshold: 触发阈值
        :param trigger_window: 触发窗口大小
        :return: 推理结果字典
        """
        print(f"Processing video: {video_path}")
        
        # 1. 特征提取
        print("Step 1: Extracting video features...")
        features = self.video_processor.extract_video_features(video_path)
        
        # 2. AICL模型推理
        print("Step 2: Running AICL model inference...")
        outputs = self.aicl_model.predict(features)
        
        actionness1 = outputs['actionness1']
        actionness2 = outputs['actionness2']
        
        # 3. 时间片段裁剪（合并连续高actionness区间）
        print("Step 3: Extracting temporal proposals...")
        temporal_proposals = self.video_processor.temporal_proposal_extraction(
            actionness1, actionness2, 
            threshold=actionness_threshold, 
            min_duration=min_duration
        )
        
        print(f"Found {len(temporal_proposals)} temporal proposals")
        
        # 4. 对高置信片段进行YOLO人体检测
        print("Step 4: Running YOLO human detection on high-confidence segments...")
        
        # 获取视频FPS
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 将时间片段转换为帧范围
        frame_proposals = []
        for start_t, end_t in temporal_proposals:
            start_frame = int(start_t * fps)
            end_frame = int(end_t * fps)
            frame_proposals.append((start_frame, end_frame))
        
        # 在高actionness时间段内检测人体
        yolo_results = self.yolo_detector.detect_persons_in_video_segment(
            video_path, 
            [(start/fps, end/fps) for start, end in frame_proposals], 
            fps=int(fps)
        )
        
        # 5. 对检测到的人体进行关键点检测（仅在YOLO确认存在真实人体的情况下）
        print("Step 5: Running pose estimation on confirmed human detections...")
        pose_results = {}
        
        for frame_num, detections in yolo_results.items():
            if detections:  # 如果YOLO检测到人体（避免误触发）
                # 读取特定帧
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # 在检测到的人体上进行姿态估计
                    frame_poses = self.pose_estimator.estimate_pose_in_frame(frame, detections)
                    
                    # 分析姿态特征
                    analyzed_poses = []
                    for pose in frame_poses:
                        features = self.pose_estimator.analyze_pose_features(pose)
                        pose['features'] = features
                        analyzed_poses.append(pose)
                    
                    pose_results[frame_num] = analyzed_poses
        
        # 6. 动作/违规/风险判定（仅在确认有人体的情况下）
        print("Step 6: Performing action/risk classification on confirmed detections...")
        classification_results = {}
        
        for frame_num, poses in pose_results.items():
            if poses:  # 仅在有姿态估计结果时进行分类
                # 对每一帧的姿态进行分类
                frame_classifications = []
                for pose in poses:
                    action, confidence = self.action_classifier.predict_action([pose])
                    frame_classifications.append({
                        'action': action,
                        'confidence': confidence,
                        'keypoints': pose['keypoints']
                    })
                
                classification_results[frame_num] = frame_classifications
        
        # 7. 整合结果
        results = {
            'temporal_proposals': temporal_proposals,
            'yolo_detections': yolo_results,
            'pose_estimations': pose_results,  # 仅包含确认有人体的帧
            'classifications': classification_results,  # 仅包含确认有人体的分类结果
            'actionness1': actionness1.cpu().numpy(),
            'actionness2': actionness2.cpu().numpy()
        }
        
        print("Inference completed successfully!")
        return results
    
    def visualize_results(self, 
                         video_path: str, 
                         results: Dict, 
                         output_path: str = None) -> str:
        """
        可视化结果
        :param video_path: 原始视频路径
        :param results: 推理结果
        :param output_path: 输出视频路径
        :return: 输出视频路径
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{base_name}_annotated.mp4"
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检查当前帧是否有检测结果
            if frame_num in results['yolo_detections']:
                detections = results['yolo_detections'][frame_num]
                
                # 绘制YOLO检测框
                for detection in detections:
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # 绘制人体边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制姿态关键点（仅在确认有人体时）
            if frame_num in results['pose_estimations']:
                poses = results['pose_estimations'][frame_num]
                
                for pose in poses:
                    keypoints = pose['keypoints']
                    
                    # 绘制关键点
                    for kp in keypoints:
                        x, y = int(kp['x']), int(kp['y'])
                        conf = kp['confidence']
                        
                        if conf > 0.5:  # 只绘制置信度高的关键点
                            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
                    
                    # 绘制骨架连接
                    self._draw_skeleton(frame, keypoints)
            
            # 添加分类结果文本（仅在确认有人体时）
            if frame_num in results['classifications']:
                classifications = results['classifications'][frame_num]
                
                for i, cls_info in enumerate(classifications):
                    action = cls_info['action']
                    conf = cls_info['confidence']
                    
                    text = f"Action: {action} ({conf:.2f})"
                    cv2.putText(frame, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            frame_num += 1
        
        cap.release()
        out.release()
        
        print(f"Annotated video saved to: {output_path}")
        return output_path
    
    def _draw_skeleton(self, frame, keypoints: List[Dict]):
        """绘制人体骨架连接线"""
        # COCO格式的关键点连接关系
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], 
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], 
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], 
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # 将关键点按索引组织
        kp_dict = {}
        for i, kp in enumerate(keypoints):
            if kp['confidence'] > 0.5:
                kp_dict[i] = (int(kp['x']), int(kp['y']))
        
        # 绘制骨架连接
        for connection in skeleton:
            kp1_idx, kp2_idx = connection[0]-1, connection[1]-1  # COCO索引从1开始
            
            if kp1_idx in kp_dict and kp2_idx in kp_dict:
                pt1 = kp_dict[kp1_idx]
                pt2 = kp_dict[kp2_idx]
                
                cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
    
    def run_trigger_detection(self, 
                            video_path: str, 
                            trigger_threshold: float = 0.6,
                            trigger_window: int = 16) -> bool:
        """
        运行触发检测：当actionness满足条件时激活下游检测
        :param video_path: 视频路径
        :param trigger_threshold: 触发阈值
        :param trigger_window: 触发窗口大小
        :return: 是否触发
        """
        # 提取特征
        features = self.video_processor.extract_video_features(video_path)
        
        # 获取actionness分数
        actionness1, actionness2 = self.aicl_model.get_actionness_scores(features)
        
        # 合并actionness
        combined_actionness = (actionness1 + actionness2) / 2.0
        
        # 检查是否触发
        trigger = self.video_processor.temporal_trigger(
            combined_actionness, 
            threshold=trigger_threshold, 
            window_size=trigger_window
        )
        
        return trigger


def run_application(video_path: str, 
                   model_path: str = None, 
                   config=None,
                   output_video: str = None,
                   actionness_threshold: float = 0.5):
    """
    运行完整的AICL应用程序
    :param video_path: 输入视频路径
    :param model_path: 模型权重路径
    :param config: 配置对象
    :param output_video: 输出视频路径
    :param actionness_threshold: actionness阈值
    """
    # 创建流水线实例
    pipeline = AICLPipeline(
        model_path=model_path,
        config=config
    )
    
    # 运行推理
    results = pipeline.run_inference(
        video_path=video_path,
        actionness_threshold=actionness_threshold
    )
    
    # 可视化结果
    annotated_video_path = pipeline.visualize_results(
        video_path=video_path,
        results=results,
        output_path=output_video
    )
    
    # 打印摘要统计
    print("\n=== Inference Results Summary ===")
    print(f"Number of temporal proposals: {len(results['temporal_proposals'])}")
    print(f"Number of frames with detections: {len(results['yolo_detections'])}")
    print(f"Total poses estimated: {sum(len(poses) for poses in results['pose_estimations'].values())}")
    
    detected_actions = {}
    for frame_classifications in results['classifications'].values():
        for cls_info in frame_classifications:
            action = cls_info['action']
            if action not in detected_actions:
                detected_actions[action] = 0
            detected_actions[action] += 1
    
    print(f"Detected actions: {detected_actions}")
    
    return results, annotated_video_path


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='AICL Video Analysis Pipeline')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--model_path', type=str, default=None, help='Path to AICL model weights')
    parser.add_argument('--output_path', type=str, default=None, help='Path for output video')
    parser.add_argument('--threshold', type=float, default=0.5, help='Actionness threshold')
    
    args = parser.parse_args()
    
    # 运行应用程序
    results, output_video = run_application(
        video_path=args.video_path,
        model_path=args.model_path,
        output_video=args.output_path,
        actionness_threshold=args.threshold
    )
    
    print(f"Processing complete! Output saved to: {output_video}")