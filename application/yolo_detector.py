"""
YOLO人体检测器
用于在高actionness时间段内检测人体
"""
import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO
import os


class YOLODetector:
    """YOLO人体检测器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化YOLO检测器
        :param model_path: YOLO模型路径或模型名称，如果为None则尝试使用output目录中的模型
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 如果没有指定模型路径，尝试使用output目录中的模型
        if model_path is None:
            # 查找output目录中的YOLO模型
            possible_paths = [
                '../output/BeltDetection(yolo11n).pt',  # 您提供的模型之一
                '../output/yolov8n-pose.pt',            # 如果有这个模型
                'yolov8n.pt'                           # 默认模型
            ]
            
            model_path = 'yolov8n.pt'  # 默认值
            for path in possible_paths:
                full_path = os.path.join(os.path.dirname(__file__), path)
                if os.path.exists(full_path):
                    model_path = full_path
                    break
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # 人体类别ID (COCO数据集中人的类别ID是0)
        self.person_class_id = 0
    
    def detect_persons_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        在单帧中检测人体
        :param frame: 输入图像帧
        :return: 检测结果列表，每个元素包含bbox、置信度等信息
        """
        # 运行YOLO检测
        results = self.model(frame, classes=[self.person_class_id], device=self.device)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 获取边界框坐标 (x1, y1, x2, y2)
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    detection = {
                        'bbox': xyxy,
                        'confidence': conf,
                        'class_id': int(box.cls[0].cpu().numpy()),
                        'center': ((xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2)
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_persons_in_video_segment(self, 
                                      video_path: str, 
                                      time_segments: List[Tuple[float, float]], 
                                      fps: int = 30) -> Dict[int, List]:
        """
        在视频的时间段内检测人体
        :param video_path: 视频路径
        :param time_segments: 时间段列表 [(start_time, end_time), ...]
        :param fps: 视频帧率
        :return: 每个时间段的检测结果
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        frame_results = {}
        
        for start_time, end_time in time_segments:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            segment_detections = []
            
            # 设置视频位置到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 在当前帧中检测人体
                detections = self.detect_persons_in_frame(frame)
                
                if detections:  # 只保存有检测结果的帧
                    frame_results[current_frame] = detections
                    segment_detections.append({
                        'frame_number': current_frame,
                        'detections': detections
                    })
                
                current_frame += 1
            
            # 为每个时间段保存检测结果
            if start_frame not in frame_results:
                frame_results[start_frame] = []
        
        cap.release()
        return frame_results
    
    def filter_detections_by_actionness(self, 
                                      frame_detections: Dict[int, List],
                                      actionness_segments: List[Tuple[int, int]]) -> Dict[int, List]:
        """
        根据actionness时间段过滤检测结果
        :param frame_detections: 帧检测结果
        :param actionness_segments: actionness时间段
        :return: 过滤后的检测结果
        """
        filtered_results = {}
        
        for frame_num, detections in frame_detections.items():
            # 检查帧号是否在任意actionness时间段内
            in_actionness_segment = any(
                start <= frame_num <= end 
                for start, end in actionness_segments
            )
            
            if in_actionness_segment and detections:
                filtered_results[frame_num] = detections
        
        return filtered_results
    
    def crop_person_roi(self, frame: np.ndarray, bbox: np.ndarray, margin: float = 0.1) -> np.ndarray:
        """
        根据边界框裁剪人体区域
        :param frame: 原始帧
        :param bbox: 边界框 [x1, y1, x2, y2]
        :param margin: 边距比例
        :return: 裁剪后的人体ROI
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # 添加边距
        h, w = frame.shape[:2]
        margin_x = int((x2 - x1) * margin)
        margin_y = int((y2 - y1) * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        return frame[y1:y2, x1:x2]