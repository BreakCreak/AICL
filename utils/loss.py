import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)
        return first_term + second_term


class GateConstraintLoss(nn.Module):
    """
    门控约束损失函数，评估各分支性能并对表现较差的分支施加惩罚
    """
    def __init__(self, weight=0.1):
        super(GateConstraintLoss, self).__init__()
        self.weight = weight
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, cas_main, cas_branches, labels=None):
        """
        Args:
            cas_main: 主融合分支的分类结果 [B, T, num_classes]
            cas_branches: 各单独分支的分类结果列表 [cas_rgb, cas_flow, cas_mixed1, cas_mixed2]
            labels: 真实标签（可选，用于更精确的性能评估）
        """
        batch_size, seq_len, num_classes = cas_main.shape
        
        # 计算每个分支与主融合结果的差异
        branch_losses = []
        for cas_branch in cas_branches:
            # 计算分支与主结果的MSE损失
            branch_diff = self.mse_loss(cas_branch, cas_main.detach())
            # 对每个样本计算平均差异
            avg_branch_diff = branch_diff.mean(dim=[1, 2])  # [batch_size]
            branch_losses.append(avg_branch_diff)
        
        # 堆叠所有分支的性能损失
        all_branch_losses = torch.stack(branch_losses, dim=1)  # [batch_size, num_branches]
        
        # 计算性能方差作为门控约束 - 鼓励门控关注性能更好的分支
        # 性能差异越大，说明某些分支表现显著优于其他分支，应该加强这种区分
        performance_variance = torch.var(all_branch_losses, dim=1).mean()
        
        return self.weight * performance_variance


class EntropyRegularizationLoss(nn.Module):
    """
    门控权重的熵正则化损失，防止门控权重过于集中或分散
    """
    def __init__(self, weight=0.01):
        super(EntropyRegularizationLoss, self).__init__()
        self.weight = weight
        
    def forward(self, gate_weights):
        """
        Args:
            gate_weights: 门控权重 [B, 4, T] - 对应RGB, Flow, Mixed1, Mixed2
        """
        # 计算门控权重的熵
        gate_weights = torch.clamp(gate_weights, min=1e-6, max=1.0-1e-6)  # 避免log(0)
        entropy = -(gate_weights * torch.log(gate_weights)).sum(dim=1).mean()
        return self.weight * entropy
