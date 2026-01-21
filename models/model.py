import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import math
import numpy as np
torch.set_printoptions(profile="full")

class BaseModel(nn.Module):
    def __init__(self, len_feature, num_classes, config=None):
        super(BaseModel, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.config = config

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=self.num_classes, kernel_size=1, padding=0),
        )

        self.action_module_rgb = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls_rgb = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.action_module_flow = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls_flow = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)
        
        # 新增两个混合分支
        self.action_module_mixed1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_mixed1 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.action_module_mixed2 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature // 2, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cls_mixed2 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(p=0.5)  # 0.5

        # 多尺度时间卷积
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(512, 512, 3, padding=1),
            nn.Conv1d(512, 512, 5, padding=2),
            nn.Conv1d(512, 512, 7, padding=3)
        ])
        
        # 多尺度特征降维
        self.reduce_dim = nn.Conv1d(512 * 3, 512, 1)  # 3个卷积层的输出拼接后降维
        
        # Actionness Head
        self.actionness_head = nn.Sequential(
            nn.Conv1d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 1, 1),
            nn.Sigmoid()
        )

        # 修改门控模块为4个通道，对应4个分支
        self.gate_module = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=4, kernel_size=1),  # 4个通道分别表示4个分支的权重
            nn.Softmax(dim=1)  # 沿着通道维度做Softmax，得到权重
        )

    def forward(self, x):
        input = x.permute(0, 2, 1)

        # 提取RGB和Flow的特征
        emb_flow = self.action_module_flow(input[:, 1024:, :])
        emb_rgb = self.action_module_rgb(input[:, :1024, :])

        # 提取混合分支的特征
        emb_mixed1 = self.action_module_mixed1(0.25 * input[:, :1024, :] + 0.75 * input[:, 1024:, :])
        emb_mixed2 = self.action_module_mixed2(0.75 * input[:, :1024, :] + 0.25 * input[:, 1024:, :])

        # 获取门控权重
        gate_weights = self.gate_module(input)  # shape: [B, 4, T]
        rgb_weight = gate_weights[:, 0:1, :]  # shape: [B, 1, T]
        flow_weight = gate_weights[:, 1:2, :]  # shape: [B, 1, T]
        mixed1_weight = gate_weights[:, 2:3, :]  # shape: [B, 1, T]
        mixed2_weight = gate_weights[:, 3:4, :]  # shape: [B, 1, T]

        # 对4个分支进行加权融合
        emb = (0.5 * rgb_weight * emb_rgb +
               0.5 * flow_weight * emb_flow +
               0.75 * mixed1_weight * emb_mixed1 +
               0.25 * mixed2_weight * emb_mixed2)

        # 多尺度时间卷积
        feat_t = emb.transpose(1, 2)  # [B, D, T] -> [B, T, D]
        feat_ms = torch.cat([conv(feat_t) for conv in self.temporal_convs], dim=1)  # [B, D_out*3, T]
        feat_reduced = self.reduce_dim(feat_ms)  # [B, D, T]
        emb_enhanced = feat_reduced.transpose(1, 2)  # [B, T, D]

        embedding_flow = emb_flow.permute(0, 2, 1)
        embedding_rgb = emb_rgb.permute(0, 2, 1)
        embedding = emb_enhanced  # 使用增强后的特征

        # 分类输出
        cas = self.cls(emb_enhanced).permute(0, 2, 1)
        actionness1 = cas.sum(dim=2)
        actionness1 = torch.sigmoid(actionness1)

        # 单独计算各分支的动作性
        action_flow = torch.sigmoid(self.cls_flow(emb_flow))
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb))
        action_mixed1 = torch.sigmoid(self.cls_mixed1(emb_mixed1))
        action_mixed2 = torch.sigmoid(self.cls_mixed2(emb_mixed2))

        action_flow = action_flow.squeeze(1)
        action_rgb = action_rgb.squeeze(1)
        action_mixed1 = action_mixed1.squeeze(1)
        action_mixed2 = action_mixed2.squeeze(1)

        actionness2 = (action_flow + action_rgb + action_mixed1 + action_mixed2) / 4
        
        # 计算actionness
        emb_t = emb.transpose(1, 2)  # [B, D, T]
        actionness = self.actionness_head(emb_t).squeeze(1)  # [B, T]

        return cas, actionness, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb


class AICL(nn.Module):
    def __init__(self, cfg):
        super(AICL, self).__init__()
        self.len_feature = 2048
        self.num_classes = 20

        self.actionness_module = BaseModel(self.len_feature, self.num_classes, cfg)

        self.softmax = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

        self.r_C = 20
        self.r_I = 20

        self.dropout = nn.Dropout(p=0.6)

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def consistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness, embeddings, k_easy):

        x = aness_bin1 + aness_bin2
        select_idx_act = actionness.new_tensor(np.where(x == 2, 1, 0))
        # print(torch.min(torch.sum(select_idx_act, dim=-1)))

        # 引入actionness gating: pos = consistent_snippets & (actionness > 0.6)
        actionness_gate = (actionness > 0.6).float()
        actionness_act = actionness * select_idx_act * actionness_gate

        select_idx_bg = actionness.new_tensor(np.where(x == 0, 1, 0))

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        # 引入actionness gating: neg = inconsistent_snippets | (actionness < 0.2)
        bg_actionness_gate = (actionness < 0.2).float()
        actionness_bg = actionness_rev * select_idx_bg + bg_actionness_gate * select_idx_bg

        easy_act = self.select_topk_embeddings(actionness_act, embeddings, k_easy)
        easy_bkg = self.select_topk_embeddings(actionness_bg, embeddings, k_easy)


        return easy_act, easy_bkg

    def Inconsistency_snippets_mining1(self, aness_bin1, aness_bin2, actionness, embeddings, k_hard):

        x = aness_bin1 + aness_bin2
        idx_region_inner = actionness.new_tensor(np.where(x == 1, 1, 0))
        # 引入actionness gating
        actionness_gate = (actionness > 0.6).float()
        aness_region_inner = actionness * idx_region_inner * actionness_gate
        hard_act = self.select_topk_embeddings(aness_region_inner, embeddings, k_hard)

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        # 引入actionness gating
        bg_actionness_gate = (actionness < 0.2).float()
        aness_region_outer = actionness_rev * idx_region_inner + bg_actionness_gate * idx_region_inner
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return hard_act, hard_bkg

    def forward(self, x):
        num_segments = x.shape[1]
        k_C = num_segments // self.r_C
        k_I = num_segments // self.r_I

        cas, actionness, action_flow, action_rgb, actionness1, actionness2, embedding, embedding_flow, embedding_rgb = self.actionness_module(x)

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_median1 = np.median(aness_np1, 1, keepdims=True)
        aness_bin1 = np.where(aness_np1 > aness_median1, 1.0, 0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_median2 = np.median(aness_np2, 1, keepdims=True)
        aness_bin2 = np.where(aness_np2 > aness_median2, 1.0, 0.0)

        # actionness = actionness1 + actionness2

        CA, CB = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        CAr, CBr = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        CAf, CBf = self.consistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.Inconsistency_snippets_mining1(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        contrast_pairs = {
            'CA': CA,
            'CB': CB,
            'IA': IA,
            'IB': IB
        }

        contrast_pairs_r = {
            'CA': CAr,
            'CB': CBr,
            'IA': IAr,
            'IB': IBr
        }

        contrast_pairs_f = {
            'CA': CAf,
            'CB': CBf,
            'IA': IAf,
            'IB': IBf
        }

        return cas, action_flow, action_rgb, contrast_pairs,contrast_pairs_r,contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2