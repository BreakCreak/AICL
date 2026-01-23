"""
视频处理器模块
负责视频预处理、特征提取等功能
"""
import cv2
import numpy as np
import torch
from typing import Tuple, List


class VideoProcessor:
    """视频处理器，负责视频读取和预处理"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def extract_video_features(self, video_path: str, feature_extractor=None) -> torch.Tensor:
        """
        提取视频特征
        :param video_path: 视频路径
        :param feature_extractor: 特征提取器（I3D/SLOWFAST/ViT等）
        :return: 特征张量
        """
        # 这里需要根据具体使用的特征提取器来实现
        # 示例：使用预训练的I3D模型提取特征
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # 读取视频帧
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 调整帧尺寸以适应特征提取器
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        
        cap.release()
        
        # 将帧转换为张量并提取特征
        if feature_extractor is not None:
            # 使用指定的特征提取器
            features = self._extract_with_model(np.array(frames), feature_extractor)
        else:
            # 模拟特征提取过程
            # 实际应用中需要使用真实的特征提取器
            num_segments = len(frames) // 16  # 假设每16帧作为一个片段
            features = torch.randn(1, num_segments, 2048)  # 模拟特征
        
        return features.to(self.device)
    
    def _extract_with_model(self, frames: np.ndarray, model) -> torch.Tensor:
        """
        使用模型提取特征
        :param frames: 视频帧数组
        :param model: 特征提取模型
        :return: 特征张量
        """
        # 实现具体的特征提取逻辑
        # 这里只是一个框架，实际实现会依赖具体模型
        with torch.no_grad():
            model.eval()
            frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0)  # [B, T, C, H, W]
            frames_tensor = frames_tensor.to(self.device)
            
            # 特征提取
            features = model(frames_tensor)
            return features
    
    def temporal_proposal_extraction(self, 
                                   actionness1: torch.Tensor, 
                                   actionness2: torch.Tensor,
                                   threshold: float = 0.5,
                                   min_duration: int = 16) -> List[Tuple[int, int]]:
        """
        从actionness序列中提取时间片段提案
        :param actionness1: 第一个actionness序列
        :param actionness2: 第二个actionness序列
        :param threshold: 阈值
        :param min_duration: 最小持续时间
        :return: 时间片段列表 [(start, end), ...]
        """
        # 合并两个actionness序列
        combined_actionness = (actionness1 + actionness2) / 2.0
        
        # 应用阈值
        high_actionness_mask = combined_actionness > threshold
        
        # 找到连续的高actionness区间
        proposals = []
        start_idx = None
        
        for i, is_high in enumerate(high_actionness_mask.squeeze()):
            if is_high and start_idx is None:
                start_idx = i
            elif not is_high and start_idx is not None:
                # 结束当前区间
                duration = i - start_idx
                if duration >= min_duration:
                    proposals.append((start_idx, i))
                start_idx = None
        
        # 处理最后一个区间
        if start_idx is not None:
            duration = len(high_actionness_mask) - start_idx
            if duration >= min_duration:
                proposals.append((start_idx, len(high_actionness_mask)))
        
        return proposals
    
    def temporal_trigger(self, 
                        actionness: torch.Tensor, 
                        threshold: float = 0.6, 
                        window_size: int = 16) -> bool:
        """
        时间触发器：当actionness在连续window_size帧上平均值超过threshold时激活
        :param actionness: actionness序列
        :param threshold: 触发阈值
        :param window_size: 滑动窗口大小
        :return: 是否触发
        """
        if len(actionness) < window_size:
            return False
        
        # 计算滑动窗口内的平均值
        for i in range(len(actionness) - window_size + 1):
            window_mean = torch.mean(actionness[i:i + window_size])
            if window_mean > threshold:
                return True
        
        return False