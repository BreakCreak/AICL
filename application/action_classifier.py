"""
动作/违规/风险分类器
基于姿态特征进行最终判定
"""
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')


class ActionClassifier:
    """动作/违规/风险分类器"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # 定义动作类别
        self.action_classes = {
            0: 'normal',
            1: 'falling',
            2: 'fighting',
            3: 'intrusion',
            4: 'fire',
            5: 'crowd_gathering'
        }
        
        # 特征名称
        self.feature_names = [
            'shoulder_angle', 'elbow_angle_left', 'elbow_angle_right',
            'knee_angle_left', 'knee_angle_right', 'center_of_gravity_x', 'center_of_gravity_y',
            'body_orientation', 'movement_speed', 'posture_stability'
        ]
    
    def extract_features_from_pose_sequence(self, pose_sequence: List[Dict]) -> np.ndarray:
        """
        从姿态序列中提取特征
        :param pose_sequence: 姿态序列
        :return: 特征向量
        """
        if not pose_sequence:
            # 返回零向量作为默认特征
            return np.zeros(len(self.feature_names))
        
        # 计算平均姿态特征
        avg_features = {}
        for feature_name in ['shoulder_angle', 'elbow_angle_left', 'elbow_angle_right', 
                           'knee_angle_left', 'knee_angle_right']:
            values = [pose.get('features', {}).get(feature_name, 0) for pose in pose_sequence 
                     if isinstance(pose, dict) and 'features' in pose]
            if values:
                avg_features[feature_name] = np.mean(values)
            else:
                avg_features[feature_name] = 0
        
        # 计算重心移动特征
        cog_positions = []
        for pose in pose_sequence:
            if isinstance(pose, dict) and 'features' in pose:
                cog = pose['features'].get('center_of_gravity', (0, 0))
                if isinstance(cog, tuple) and len(cog) == 2:
                    cog_positions.append(cog)
        
        if len(cog_positions) > 1:
            # 计算重心移动速度
            movement_vectors = []
            for i in range(1, len(cog_positions)):
                dx = cog_positions[i][0] - cog_positions[i-1][0]
                dy = cog_positions[i][1] - cog_positions[i-1][1]
                movement_vectors.append(np.sqrt(dx**2 + dy**2))
            
            movement_speed = np.mean(movement_vectors) if movement_vectors else 0
        else:
            movement_speed = 0
        
        # 计算姿势稳定性（关键点置信度变化）
        confidences = []
        for pose in pose_sequence:
            if isinstance(pose, dict) and 'keypoints' in pose:
                kp_confs = [kp.get('confidence', 0) for kp in pose['keypoints']]
                if kp_confs:
                    confidences.append(np.mean(kp_confs))
        
        posture_stability = np.std(confidences) if confidences else 0
        
        # 构建特征向量
        feature_vector = np.array([
            avg_features.get('shoulder_angle', 0),
            avg_features.get('elbow_angle_left', 0),
            avg_features.get('elbow_angle_right', 0),
            avg_features.get('knee_angle_left', 0),
            avg_features.get('knee_angle_right', 0),
            cog_positions[0][0] if cog_positions else 0,  # center_of_gravity_x
            cog_positions[0][1] if cog_positions else 0,  # center_of_gravity_y
            0,  # body_orientation (placeholder)
            movement_speed,
            posture_stability
        ])
        
        return feature_vector
    
    def predict_action(self, pose_sequence: List[Dict]) -> Tuple[str, float]:
        """
        预测动作类别
        :param pose_sequence: 姿态序列
        :return: (动作类别, 置信度)
        """
        if not self.is_trained:
            # 如果模型未训练，使用规则基础的简单判断
            return self._rule_based_prediction(pose_sequence)
        
        features = self.extract_features_from_pose_sequence(pose_sequence)
        prediction = self.model.predict([features])[0]
        probabilities = self.model.predict_proba([features])[0]
        confidence = np.max(probabilities)
        
        action_label = self.action_classes.get(prediction, 'unknown')
        return action_label, confidence
    
    def _rule_based_prediction(self, pose_sequence: List[Dict]) -> Tuple[str, float]:
        """
        基于规则的预测（当模型未训练时使用）
        :param pose_sequence: 姿态序列
        :return: (动作类别, 置信度)
        """
        if not pose_sequence:
            return 'normal', 0.8
        
        # 简单的规则基础判断
        for pose in pose_sequence:
            if isinstance(pose, dict) and 'features' in pose:
                features = pose['features']
                
                # 检测摔倒：膝盖角度异常小
                if (features.get('knee_angle_left', 180) < 45 or 
                    features.get('knee_angle_right', 180) < 45):
                    return 'falling', 0.7
                
                # 检测异常姿势：肩膀角度异常
                if features.get('shoulder_angle', 180) < 90:
                    return 'falling', 0.6
        
        return 'normal', 0.8
    
    def train(self, training_data: List[Tuple[List[Dict], int]]):
        """
        训练分类器
        :param training_data: 训练数据 [(pose_sequence, label), ...]
        """
        X, y = [], []
        for pose_seq, label in training_data:
            features = self.extract_features_from_pose_sequence(pose_seq)
            X.append(features)
            y.append(label)
        
        if X and y:
            X = np.array(X)
            y = np.array(y)
            self.model.fit(X, y)
            self.is_trained = True
            print("Action classifier trained successfully!")
        else:
            print("Warning: No training data provided")
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'action_classes': self.action_classes,
                'is_trained': self.is_trained
            }, filepath)
            print(f"Model saved to {filepath}")
        else:
            print("Warning: Model is not trained, nothing to save")
    
    def load_model(self, filepath: str):
        """加载模型"""
        loaded_data = joblib.load(filepath)
        self.model = loaded_data['model']
        self.action_classes = loaded_data['action_classes']
        self.is_trained = loaded_data['is_trained']
        print(f"Model loaded from {filepath}")