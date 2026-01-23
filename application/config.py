"""
AICL应用配置文件
定义应用的各种配置参数
"""
import os


class AppConfig:
    """AICL应用配置类"""
    
    def __init__(self):
        # 设备配置
        self.device = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'true').lower() == 'true' else 'cpu'
        
        # 模型配置
        self.aicl_model_path = None  # AICL模型权重路径
        self.yolo_model_path = 'yolov8n.pt'  # YOLO模型路径
        self.pose_model_path = 'yolov8n-pose.pt'  # 姿态估计模型路径
        
        # 推理参数
        self.actionness_threshold = 0.5  # Actionness阈值
        self.trigger_threshold = 0.6     # 触发阈值
        self.trigger_window = 16         # 触发窗口大小
        self.min_duration = 16           # 最小持续时间（帧）
        
        # 视频处理参数
        self.feature_fps = 25            # 特征提取的帧率
        self.target_height = 224         # 目标高度
        self.target_width = 224          # 目标宽度
        
        # 输出参数
        self.save_annotated_video = True  # 是否保存标注视频
        self.save_detection_results = True  # 是否保存检测结果
        self.output_dir = './output'      # 输出目录
        
        # 分类参数
        self.enable_action_classification = True  # 是否启用动作分类
        self.enable_risk_detection = True        # 是否启用风险检测
        self.enable_violation_detection = True   # 是否启用违规检测


def get_default_config():
    """获取默认配置"""
    return AppConfig()


def load_config_from_file(config_path: str):
    """从文件加载配置"""
    # 这里可以实现从JSON/YAML文件加载配置
    # 暂时返回默认配置
    return get_default_config()