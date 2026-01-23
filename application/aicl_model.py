"""
AICL模型接口
封装已训练的AICL模型进行推理
"""
import torch
import torch.nn as nn
from models.model import AICL
from typing import Tuple, Dict, Any
import os


class AICLInference:
    """AICL模型推理接口"""
    
    def __init__(self, model_path: str = None, config=None):
        """
        初始化AICL推理模型
        :param model_path: 模型权重路径
        :param config: 配置对象
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        # 创建模型实例
        self.model = AICL(config)
        self.model = self.model.to(self.device)
        
        # 加载预训练权重
        if model_path is not None:
            self.load_weights(model_path)
        else:
            # 尝试从output目录加载默认模型
            default_model_paths = [
                '../output/CAS_Only.pkl',
                '../output/model_rgb.pth',
                '../output/model_flow.pth'
            ]
            for path in default_model_paths:
                full_path = os.path.join(os.path.dirname(__file__), path)
                if os.path.exists(full_path):
                    self.load_weights(full_path)
                    break
        
        self.model.eval()
    
    def load_weights(self, model_path: str):
        """加载模型权重"""
        try:
            # 检查文件扩展名以决定如何加载模型
            if model_path.endswith('.pkl'):
                # 对于pickle文件，需要特别处理
                checkpoint = torch.load(model_path, map_location=self.device)
                # 根据模型保存方式适配加载逻辑
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint, strict=False)
                else:
                    # 如果整个文件就是状态字典
                    self.model.load_state_dict(checkpoint, strict=False)
            else:
                # 其他.pth文件
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif isinstance(checkpoint, dict):
                    self.model.load_state_dict(checkpoint, strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Failed to load model weights from {model_path}: {e}")
            print("Creating model without loading weights...")
    
    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        执行模型预测
        :param features: 输入特征 [batch_size, num_segments, feature_dim]
        :return: 包含各种输出的字典
        """
        with torch.no_grad():
            features = features.to(self.device)
            
            # 前向传播
            cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, \
            actionness1, actionness2, aness_bin1, aness_bin2, gate_weights = self.model(features)
            
            return {
                'cas': cas,
                'action_flow': action_flow,
                'action_rgb': action_rgb,
                'actionness1': actionness1,  # CAS-derived
                'actionness2': actionness2,  # RGB + Flow
                'aness_bin1': aness_bin1,
                'aness_bin2': aness_bin2,
                'contrast_pairs': contrast_pairs,
                'gate_weights': gate_weights
            }
    
    def get_actionness_scores(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取actionness分数
        :param features: 输入特征
        :return: (actionness1, actionness2)
        """
        outputs = self.predict(features)
        return outputs['actionness1'], outputs['actionness2']
    
    def get_temporal_proposals(self, 
                             features: torch.Tensor, 
                             threshold: float = 0.5,
                             min_duration: int = 16) -> list:
        """
        获取时间片段提案
        :param features: 输入特征
        :param threshold: 阈值
        :param min_duration: 最小持续时间
        :return: 时间片段列表
        """
        from .video_processor import VideoProcessor
        processor = VideoProcessor()
        
        actionness1, actionness2 = self.get_actionness_scores(features)
        return processor.temporal_proposal_extraction(
            actionness1, 
            actionness2, 
            threshold=threshold, 
            min_duration=min_duration
        )