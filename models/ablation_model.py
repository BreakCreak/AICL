"""
消融实验模型变体
(a) Single-Expert: 单一专家（RGB+Flow 简单拼接），无混合、无门控
(b) Multi-Expert (No Mixed): RGB + Flow 双专家，无混合专家
(c) Multi-Expert + Mixed (No Gate): 引入 Mixed1 / Mixed2，但使用固定权重融合
(d) Full AICL-MoE: 四专家 + 门控机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_printoptions(profile="full")


class SingleExpertModel(nn.Module):
    """(a) Single-Expert: 单一专家（RGB+Flow 简单拼接），无混合、无门控"""
    def __init__(self, len_feature, num_classes, config=None):
        super(SingleExpertModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        # 简单的单一分支
        self.action_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 分类头
        self.cls = nn.Conv1d(512, self.num_classes, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        # 简单处理整个输入
        emb = self.action_module(input)

        # 分类
        cas = self.cls(emb).permute(0, 2, 1)
        actionness = torch.sigmoid(cas.sum(dim=2))

        # 返回统一的格式
        return cas, None, None, {}, {}, {}, actionness, actionness, None, None, None


class MultiExpertNoMixedModel(nn.Module):
    """(b) Multi-Expert (No Mixed): RGB + Flow 双专家，无混合专家"""
    def __init__(self, len_feature, num_classes, config=None):
        super(MultiExpertNoMixedModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        # RGB & Flow 分支
        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_rgb = nn.Conv1d(512, 1, 1)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_flow = nn.Conv1d(512, 1, 1)

        # 固定权重融合
        self.cls = nn.Conv1d(512, self.num_classes, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        # RGB & Flow 特征
        emb_rgb = self.action_module_rgb(input[:, :1024, :])
        emb_flow = self.action_module_flow(input[:, 1024:, :])

        # 固定权重融合
        emb = 0.5 * emb_rgb + 0.5 * emb_flow

        # 分类
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = torch.sigmoid(cas.sum(dim=2))

        # 单分支 actionness
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb)).squeeze(1)
        action_flow = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)
        
        # 融合 actionness2
        actionness2 = (action_rgb + action_flow) / 2

        # 返回统一的格式
        return cas, action_flow, action_rgb, {}, {}, {}, actionness1, actionness2, None, None, None


class MultiExpertMixedNoGateModel(nn.Module):
    """(c) Multi-Expert + Mixed (No Gate): 引入 Mixed1 / Mixed2，但使用固定权重融合"""
    def __init__(self, len_feature, num_classes, config=None):
        super(MultiExpertMixedNoGateModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        # RGB & Flow 分支
        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_rgb = nn.Conv1d(512, 1, 1)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_flow = nn.Conv1d(512, 1, 1)

        # 混合分支
        self.action_module_mixed1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_mixed1 = nn.Conv1d(512, 1, 1)

        self.action_module_mixed2 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_mixed2 = nn.Conv1d(512, 1, 1)

        # 最终分类
        self.cls = nn.Conv1d(512, self.num_classes, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        # RGB & Flow 特征
        emb_rgb = self.action_module_rgb(input[:, :1024, :])
        emb_flow = self.action_module_flow(input[:, 1024:, :])

        # 混合分支
        emb_mixed1 = self.action_module_mixed1(0.25 * input[:, :1024, :] + 0.75 * input[:, 1024:, :])
        emb_mixed2 = self.action_module_mixed2(0.75 * input[:, :1024, :] + 0.25 * input[:, 1024:, :])

        # 固定权重融合
        emb = 0.25 * emb_rgb + 0.25 * emb_flow + 0.25 * emb_mixed1 + 0.25 * emb_mixed2

        # 分类
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = torch.sigmoid(cas.sum(dim=2))

        # 单分支 actionness
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb)).squeeze(1)
        action_flow = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)
        action_mixed1 = torch.sigmoid(self.cls_mixed1(emb_mixed1)).squeeze(1)
        action_mixed2 = torch.sigmoid(self.cls_mixed2(emb_mixed2)).squeeze(1)

        # 融合 actionness2
        actionness2 = (action_rgb + action_flow + action_mixed1 + action_mixed2) / 4

        # 返回统一的格式
        return cas, action_flow, action_rgb, {}, {}, {}, actionness1, actionness2, None, None, None


class FullAICLMoEModel(nn.Module):
    """(d) Full AICL-MoE: 四专家 + 门控机制"""
    def __init__(self, len_feature, num_classes, config=None):
        super(FullAICLMoEModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        # RGB & Flow 分支
        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_rgb = nn.Conv1d(512, 1, 1)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_flow = nn.Conv1d(512, 1, 1)

        # 混合分支
        self.action_module_mixed1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_mixed1 = nn.Conv1d(512, 1, 1)

        self.action_module_mixed2 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_mixed2 = nn.Conv1d(512, 1, 1)

        # 门控模块
        self.gate_module = nn.Sequential(
            nn.Conv1d(len_feature, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(512, 4, 1),
            nn.Softmax(dim=1)
        )

        # 最终分类
        self.cls = nn.Conv1d(512, self.num_classes, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, inference=False):
        input = x.permute(0, 2, 1)

        # RGB & Flow 特征
        emb_rgb = self.action_module_rgb(input[:, :1024, :])
        emb_flow = self.action_module_flow(input[:, 1024:, :])

        # 混合分支
        emb_mixed1 = self.action_module_mixed1(0.25 * input[:, :1024, :] + 0.75 * input[:, 1024:, :])
        emb_mixed2 = self.action_module_mixed2(0.75 * input[:, :1024, :] + 0.25 * input[:, 1024:, :])

        # 门控
        gate_weights = self.gate_module(input)
        if inference:
            # 软化门控
            gate_weights = gate_weights * 0.7 + 0.3 / 4

        rgb_w, flow_w, m1_w, m2_w = gate_weights[:, 0:1, :], gate_weights[:, 1:2, :], gate_weights[:, 2:3,
                                                                                      :], gate_weights[:, 3:4, :]

        # MoE 加权融合
        emb = 0.5 * rgb_w * emb_rgb + 0.5 * flow_w * emb_flow + 0.75 * m1_w * emb_mixed1 + 0.25 * m2_w * emb_mixed2

        # 分类
        cas = self.cls(emb).permute(0, 2, 1)
        actionness1 = torch.sigmoid(cas.sum(dim=2))

        # 单分支 actionness
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb)).squeeze(1)
        action_flow = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)
        action_mixed1 = torch.sigmoid(self.cls_mixed1(emb_mixed1)).squeeze(1)
        action_mixed2 = torch.sigmoid(self.cls_mixed2(emb_mixed2)).squeeze(1)

        # 融合 actionness2，保留低片段信息
        low_action = torch.min(action_rgb, action_flow)
        actionness2 = (0.6 * action_mixed1 + 0.4 * action_mixed2 + 0.5 * action_rgb + 0.5 * action_flow) / 4
        actionness2 = 0.8 * actionness2 + 0.2 * low_action

        # 返回统一的格式
        return cas, action_flow, action_rgb, {}, {}, {}, actionness1, actionness2, None, None, gate_weights


class AICLAblation(nn.Module):
    """消融实验主类，可以选择不同变体"""
    def __init__(self, cfg, variant='full'):
        """
        Args:
            cfg: 配置对象
            variant: 模型变体 ['single', 'multi_no_mixed', 'multi_mixed_no_gate', 'full']
        """
        super(AICLAblation, self).__init__()
        self.len_feature = 2048
        self.num_classes = 20
        self.variant = variant
        
        if variant == 'single':
            # (a) Single-Expert
            self.model = SingleExpertModel(self.len_feature, self.num_classes, cfg)
        elif variant == 'multi_no_mixed':
            # (b) Multi-Expert (No Mixed)
            self.model = MultiExpertNoMixedModel(self.len_feature, self.num_classes, cfg)
        elif variant == 'multi_mixed_no_gate':
            # (c) Multi-Expert + Mixed (No Gate)
            self.model = MultiExpertMixedNoGateModel(self.len_feature, self.num_classes, cfg)
        elif variant == 'full':
            # (d) Full AICL-MoE
            self.model = FullAICLMoEModel(self.len_feature, self.num_classes, cfg)
        else:
            raise ValueError(f"Unknown variant: {variant}")
            
        self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        # 所有变体都返回相同的接口格式
        return self.model(x)