"""
人体关键点检测器
使用OpenPose/HRNet/RTMPose等模型进行关键点检测
"""
import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict
from ultralytics import YOLO
import os


class PoseEstimator:
    """人体姿态估计器"""
    
    def __init__(self, model_path: str = None):
        """
        初始化姿态估计器
        :param model_path: 姿态估计模型路径，如果为None则尝试使用output目录中的模型
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 如果没有指定模型路径，尝试使用output目录中的模型
        if model_path is None:
            # 查找output目录中的姿态估计模型
            possible_paths = [
                '../output/yolov8n-pose.pt',  # 您提供的姿态估计模型
                'yolov8n-pose.pt'            # 默认模型
            ]
            
            model_path = 'yolov8n-pose.pt'  # 默认值
            for path in possible_paths:
                full_path = os.path.join(os.path.dirname(__file__), path)
                if os.path.exists(full_path):
                    model_path = full_path
                    break
        
        self.pose_model = YOLO(model_path)
        self.pose_model.to(self.device)
        
        # COCO关键点定义
        self.keypoints_def = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def estimate_pose_in_roi(self, roi: np.ndarray) -> List[Dict]:
        """
        在ROI区域内估计人体姿态
        :param roi: 人体ROI图像
        :return: 关键点检测结果
        """
        results = self.pose_model(roi, device=self.device)
        
        pose_results = []
        for result in results:
            keypoints = result.keypoints
            if keypoints is not None and len(keypoints) > 0:
                for person_keypoints in keypoints:
                    # 获取关键点坐标和置信度
                    xy_conf = person_keypoints.data[0].cpu().numpy()  # [x, y, confidence]
                    
                    person_pose = {
                        'keypoints': [],
                        'bbox': self._calculate_bbox_from_keypoints(xy_conf),
                        'visibility': self._calculate_visibility(xy_conf)
                    }
                    
                    for i, (x, y, conf) in enumerate(xy_conf):
                        keypoint_info = {
                            'name': self.keypoints_def[i] if i < len(self.keypoints_def) else f'keypoint_{i}',
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf)
                        }
                        person_pose['keypoints'].append(keypoint_info)
                    
                    pose_results.append(person_pose)
        
        return pose_results
    
    def estimate_pose_in_frame(self, frame: np.ndarray, detections: List[Dict] = None) -> List[Dict]:
        """
        在整个帧中估计姿态（可选地仅在检测到的人体区域）
        :param frame: 输入帧
        :param detections: 人体检测结果（如果提供，则仅在检测到的区域内估计姿态）
        :return: 姿态估计结果
        """
        if detections is not None and len(detections) > 0:
            # 仅在检测到的人体区域内估计姿态
            results = []
            for detection in detections:
                # 裁剪人体区域
                roi = self._crop_roi(frame, detection['bbox'])
                
                # 在ROI中估计姿态
                pose_in_roi = self.estimate_pose_in_roi(roi)
                
                # 调整关键点坐标到原始帧坐标系
                adjusted_poses = self._adjust_coordinates(pose_in_roi, detection['bbox'])
                results.extend(adjusted_poses)
            
            return results
        else:
            # 在整个帧中估计姿态
            results = self.pose_model(frame, device=self.device)
            
            pose_results = []
            for result in results:
                keypoints = result.keypoints
                if keypoints is not None and len(keypoints) > 0:
                    for person_keypoints in keypoints:
                        xy_conf = person_keypoints.data[0].cpu().numpy()
                        
                        person_pose = {
                            'keypoints': [],
                            'bbox': self._calculate_bbox_from_keypoints(xy_conf),
                            'visibility': self._calculate_visibility(xy_conf)
                        }
                        
                        for i, (x, y, conf) in enumerate(xy_conf):
                            keypoint_info = {
                                'name': self.keypoints_def[i] if i < len(self.keypoints_def) else f'keypoint_{i}',
                                'x': float(x),
                                'y': float(y),
                                'confidence': float(conf)
                            }
                            person_pose['keypoints'].append(keypoint_info)
                        
                        pose_results.append(person_pose)
            
            return pose_results
    
    def _crop_roi(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """裁剪ROI区域"""
        x1, y1, x2, y2 = map(int, bbox)
        return frame[y1:y2, x1:x2]
    
    def _adjust_coordinates(self, poses: List[Dict], bbox: np.ndarray) -> List[Dict]:
        """调整关键点坐标到原始帧坐标系"""
        x1, y1, x2, y2 = bbox
        offset_x, offset_y = x1, y1
        
        adjusted_poses = []
        for pose in poses:
            adjusted_pose = {
                'keypoints': [],
                'bbox': pose['bbox'] + np.array([offset_x, offset_y, offset_x, offset_y]),
                'visibility': pose['visibility']
            }
            
            for kp in pose['keypoints']:
                adjusted_kp = {
                    'name': kp['name'],
                    'x': kp['x'] + offset_x,
                    'y': kp['y'] + offset_y,
                    'confidence': kp['confidence']
                }
                adjusted_pose['keypoints'].append(adjusted_kp)
            
            adjusted_poses.append(adjusted_pose)
        
        return adjusted_poses
    
    def _calculate_bbox_from_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """从关键点计算边界框"""
        visible_keypoints = keypoints[keypoints[:, 2] > 0.5]  # 只考虑置信度大于0.5的关键点
        if len(visible_keypoints) == 0:
            return np.array([0, 0, 0, 0])
        
        x_coords = visible_keypoints[:, 0]
        y_coords = visible_keypoints[:, 1]
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def _calculate_visibility(self, keypoints: np.ndarray) -> float:
        """计算可见关键点的比例"""
        visible_count = np.sum(keypoints[:, 2] > 0.5)  # 置信度大于0.5视为可见
        total_count = len(keypoints)
        return visible_count / total_count if total_count > 0 else 0.0
    
    def analyze_pose_features(self, pose_result: Dict) -> Dict:
        """
        分析姿态特征，用于后续的动作/违规/风险判定
        :param pose_result: 姿态估计结果
        :return: 姿态特征字典
        """
        features = {}
        
        # 计算身体各部分的角度
        keypoints = pose_result['keypoints']
        keypoint_dict = {kp['name']: kp for kp in keypoints}
        
        # 计算主要关节角度
        features['shoulder_angle'] = self._calculate_angle(
            keypoint_dict.get('left_shoulder', {}),
            keypoint_dict.get('right_shoulder', {}),
            keypoint_dict.get('left_elbow', {})
        ) if all(name in keypoint_dict for name in ['left_shoulder', 'right_shoulder', 'left_elbow']) else 0
        
        features['elbow_angle_left'] = self._calculate_angle(
            keypoint_dict.get('left_shoulder', {}),
            keypoint_dict.get('left_elbow', {}),
            keypoint_dict.get('left_wrist', {})
        ) if all(name in keypoint_dict for name in ['left_shoulder', 'left_elbow', 'left_wrist']) else 0
        
        features['elbow_angle_right'] = self._calculate_angle(
            keypoint_dict.get('right_shoulder', {}),
            keypoint_dict.get('right_elbow', {}),
            keypoint_dict.get('right_wrist', {})
        ) if all(name in keypoint_dict for name in ['right_shoulder', 'right_elbow', 'right_wrist']) else 0
        
        features['knee_angle_left'] = self._calculate_angle(
            keypoint_dict.get('left_hip', {}),
            keypoint_dict.get('left_knee', {}),
            keypoint_dict.get('left_ankle', {})
        ) if all(name in keypoint_dict for name in ['left_hip', 'left_knee', 'left_ankle']) else 0
        
        features['knee_angle_right'] = self._calculate_angle(
            keypoint_dict.get('right_hip', {}),
            keypoint_dict.get('right_knee', {}),
            keypoint_dict.get('right_ankle', {})
        ) if all(name in keypoint_dict for name in ['right_hip', 'right_knee', 'right_ankle']) else 0
        
        # 计算身体重心
        features['center_of_gravity'] = self._calculate_center_of_gravity(keypoints)
        
        return features
    
    def _calculate_angle(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        """计算三个点形成的角度"""
        import math
        
        # 将关键点转换为向量
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # 防止数值误差
        angle = math.degrees(math.acos(cos_angle))
        
        return angle
    
    def _calculate_center_of_gravity(self, keypoints: List[Dict]) -> Tuple[float, float]:
        """计算身体重心"""
        visible_keypoints = [kp for kp in keypoints if kp['confidence'] > 0.5]
        
        if not visible_keypoints:
            return (0, 0)
        
        x_coords = [kp['x'] for kp in visible_keypoints]
        y_coords = [kp['y'] for kp in visible_keypoints]
        
        cog_x = sum(x_coords) / len(visible_keypoints)
        cog_y = sum(y_coords) / len(visible_keypoints)
        
        return (cog_x, cog_y)