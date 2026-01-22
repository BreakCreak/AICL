import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_printoptions(profile="full")


# ===================== BaseModel =====================
class BaseModel(nn.Module):
    def __init__(self, len_feature, num_classes, config=None):
        super(BaseModel, self).__init__()
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
        emb = 0.5 * rgb_w * emb_rgb + 0.5 * flow_w * emb_mixed1 + 0.75 * m1_w * emb_mixed1 + 0.25 * m2_w * emb_mixed2

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

        return cas, action_flow, action_rgb, action_mixed1, action_mixed2, actionness1, actionness2, emb, emb_flow, emb_rgb, gate_weights


# ===================== AICL =====================
class AICL(nn.Module):
    def __init__(self, cfg):
        super(AICL, self).__init__()
        self.len_feature = 2048
        self.num_classes = 20
        self.r_C = 20
        self.r_I = 30  # 改为30以降低hard采样比例，防止hard negative过干
        self.model = BaseModel(self.len_feature, self.num_classes, cfg)
        self.dropout = nn.Dropout(0.6)

    def select_topk_embeddings(self, scores, embeddings, k, retain_random=0.2):
        """
        scores: [B, T]
        embeddings: [B, T, D]
        k: topk number
        retain_random: 保留低分片段比例
        """
        B, T, D = embeddings.shape
        device = embeddings.device

        # 确保scores中没有NaN或inf值
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e5, neginf=-1e5)
        
        # 根据经验教训，对k值进行边界检查，确保k不超过张量对应维度的实际大小
        actual_k = min(k, T)
        if actual_k <= 0:
            # 如果实际k为0或负数，返回空张量
            return embeddings.new_empty((B, 0, D))

        # 计算top-k和随机k的数量
        k_top = int(actual_k * (1 - retain_random))
        k_rand = actual_k - k_top

        # 确保k_top和k_rand不为负数
        k_top = max(0, min(k_top, actual_k))
        k_rand = max(0, min(k_rand, actual_k - k_top))

        # 对scores进行排序获取top-k索引
        _, sorted_indices = scores.sort(descending=True, dim=1)
        
        # 构造最终索引张量，预先分配正确大小
        final_indices = torch.zeros((B, actual_k), dtype=torch.long, device=device)
        
        for b in range(B):
            # 确保索引在有效范围内
            valid_sorted_indices = torch.clamp(sorted_indices[b, :k_top], 0, T-1)
            
            # 获取随机k的索引，确保不与top-k重复
            available_indices = []
            selected_set = set(valid_sorted_indices.cpu().numpy())
            for i in range(T):
                if i not in selected_set:
                    available_indices.append(i)
            
            # 从可用索引中随机选择
            random_indices_b = torch.empty(0, dtype=torch.long, device=device)
            if len(available_indices) > 0 and k_rand > 0:
                selected_random = np.random.choice(
                    available_indices, 
                    size=min(k_rand, len(available_indices)), 
                    replace=False
                )
                if len(selected_random) > 0:
                    random_indices_b = torch.tensor(selected_random, dtype=torch.long, device=device)
            
            # 组合top-k和随机索引
            combined_indices = torch.cat([valid_sorted_indices, random_indices_b])
            
            # 确保总长度不超过actual_k
            if combined_indices.size(0) > actual_k:
                combined_indices = combined_indices[:actual_k]
            elif combined_indices.size(0) < actual_k:
                # 如果不够，用第一个索引填充（这种情况理论上不应该发生，但为了安全性）
                if combined_indices.size(0) > 0:
                    padding = combined_indices[0].repeat(actual_k - combined_indices.size(0))
                    combined_indices = torch.cat([combined_indices, padding])
            
            # 再次确保所有索引在有效范围内
            combined_indices = torch.clamp(combined_indices, 0, T-1)
            
            # 存储到最终索引张量
            final_indices[b] = combined_indices

        # 使用gather操作提取对应的embeddings
        expanded_indices = final_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_embeddings = torch.gather(embeddings, 1, expanded_indices)
        
        return selected_embeddings

    def consistency_snippets_mining(self, aness_bin1, aness_bin2, actionness, embeddings, k_easy):
        x = aness_bin1 + aness_bin2
        select_idx_act = actionness.new_tensor(np.where(x == 2, 1, 0))
        select_idx_bg = actionness.new_tensor(np.where(x == 0, 1, 0))
        easy_act = self.select_topk_embeddings(actionness * select_idx_act, embeddings, k_easy)
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        easy_bkg = self.select_topk_embeddings(actionness_rev * select_idx_bg, embeddings, k_easy)
        return easy_act, easy_bkg

    def inconsistency_snippets_mining(self, aness_bin1, aness_bin2, actionness, embeddings, k_hard):
        x = aness_bin1 + aness_bin2
        idx_region_inner = actionness.new_tensor(np.where(x == 1, 1, 0))

        # 软门控 instead of hard 0.6 threshold
        gate_center = 0.4
        gate_scale = 5.0
        actionness_gate = torch.sigmoid((actionness - gate_center) * gate_scale)
        aness_region_inner = actionness * idx_region_inner * actionness_gate

        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard, retain_random=0.2)

        # 背景片段也软化
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        bg_gate = torch.sigmoid((0.3 - actionness) * 5)  # soft gate
        aness_region_outer = actionness_rev * idx_region_inner + bg_gate * idx_region_inner

        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard, retain_random=0.2)

        return hard_act, hard_bkg

    def forward(self, x):
        num_segments = x.shape[1]
        k_C = num_segments // self.r_C
        k_I = num_segments // self.r_I

        cas, action_flow, action_rgb, action_mixed1, action_mixed2, actionness1, actionness2, embedding, embedding_flow, embedding_rgb, gate_weights = self.model(
            x)

        aness_np1 = actionness1.cpu().detach().numpy()
        thr1 = np.percentile(aness_np1, 60, axis=1, keepdims=True)  # 使用60%分位数替代中位数
        aness_bin1 = (aness_np1 > thr1).astype(np.float32)  # 改为分位数二值化

        aness_np2 = actionness2.cpu().detach().numpy()
        thr2 = np.percentile(aness_np2, 60, axis=1, keepdims=True)  # 使用60%分位数替代中位数
        aness_bin2 = (aness_np2 > thr2).astype(np.float32)  # 改为分位数二值化

        # mining
        CA, CB = self.consistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.inconsistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        CAr, CBr = self.consistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.inconsistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        CAf, CBf = self.consistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.inconsistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        contrast_pairs = {'CA': CA, 'CB': CB, 'IA': IA, 'IB': IB}
        contrast_pairs_r = {'CA': CAr, 'CB': CBr, 'IA': IAr, 'IB': IBr}
        contrast_pairs_f = {'CA': CAf, 'CB': CBf, 'IA': IAf, 'IB': IBf}

        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights

