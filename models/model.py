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
        emb_mixed1 = self.action_module_mixed1(0.25*input[:, :1024, :] + 0.75*input[:, 1024:, :])
        emb_mixed2 = self.action_module_mixed2(0.75*input[:, :1024, :] + 0.25*input[:, 1024:, :])

        # 门控
        gate_weights = self.gate_module(input)
        if inference:
            # 软化门控
            gate_weights = gate_weights * 0.7 + 0.3 / 4

        rgb_w, flow_w, m1_w, m2_w = gate_weights[:,0:1,:], gate_weights[:,1:2,:], gate_weights[:,2:3,:], gate_weights[:,3:4,:]

        # MoE 加权融合
        emb = 0.5*rgb_w*emb_rgb + 0.5*flow_w*emb_flow + 0.75*m1_w*emb_mixed1 + 0.25*m2_w*emb_mixed2

        # 分类
        cas = self.cls(emb).permute(0,2,1)
        actionness1 = torch.sigmoid(cas.sum(dim=2))

        # 单分支 actionness
        action_rgb = torch.sigmoid(self.cls_rgb(emb_rgb)).squeeze(1)
        action_flow = torch.sigmoid(self.cls_flow(emb_flow)).squeeze(1)
        action_mixed1 = torch.sigmoid(self.cls_mixed1(emb_mixed1)).squeeze(1)
        action_mixed2 = torch.sigmoid(self.cls_mixed2(emb_mixed2)).squeeze(1)

        # 融合 actionness2，保留低片段信息
        low_action = torch.min(action_rgb, action_flow)
        actionness2 = (0.6*action_mixed1 + 0.4*action_mixed2 + 0.5*action_rgb + 0.5*action_flow)/4
        actionness2 = 0.8*actionness2 + 0.2*low_action

        return cas, action_flow, action_rgb, action_mixed1, action_mixed2, actionness1, actionness2, emb, emb_flow, emb_rgb

# ===================== AICL =====================
class AICL(nn.Module):
    def __init__(self, cfg):
        super(AICL, self).__init__()
        self.len_feature = 2048
        self.num_classes = 20
        self.r_C = 20
        self.r_I = 20
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
        
        # 排序
        _, idx_DESC = scores.sort(descending=True, dim=1)
        k_top = int(k * (1 - retain_random))
        k_rand = k - k_top

        idx_topk = idx_DESC[:, :k_top]  # top-k
        if k_rand > 0:
            rand_idx = torch.randint(0, T, (B, k_rand), device=device)
            idx_topk = torch.cat([idx_topk, rand_idx], dim=1)

        # gather
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, D])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def consistency_snippets_mining(self, aness_bin1, aness_bin2, actionness, embeddings, k_easy):
        x = aness_bin1 + aness_bin2
        select_idx_act = actionness.new_tensor(np.where(x==2,1,0))
        select_idx_bg = actionness.new_tensor(np.where(x==0,1,0))
        easy_act = self.select_topk_embeddings(actionness*select_idx_act, embeddings, k_easy)
        actionness_rev = torch.max(actionness,dim=1,keepdim=True)[0]-actionness
        easy_bkg = self.select_topk_embeddings(actionness_rev*select_idx_bg, embeddings, k_easy)
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

        cas, action_flow, action_rgb, action_mixed1, action_mixed2, actionness1, actionness2, embedding, embedding_flow, embedding_rgb = self.model(x)

        aness_np1 = actionness1.cpu().detach().numpy()
        aness_bin1 = np.where(aness_np1 > np.median(aness_np1,1,keepdims=True),1.0,0.0)

        aness_np2 = actionness2.cpu().detach().numpy()
        aness_bin2 = np.where(aness_np2 > np.median(aness_np2,1,keepdims=True),1.0,0.0)

        # mining
        CA, CB = self.consistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding, k_C)
        IA, IB = self.inconsistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding, k_I)

        CAr, CBr = self.consistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_C)
        IAr, IBr = self.inconsistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_rgb, k_I)

        CAf, CBf = self.consistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_flow, k_C)
        IAf, IBf = self.inconsistency_snippets_mining(aness_bin1, aness_bin2, actionness1, embedding_flow, k_I)

        contrast_pairs = {'CA': CA,'CB': CB,'IA': IA,'IB': IB}
        contrast_pairs_r = {'CA': CAr,'CB': CBr,'IA': IAr,'IB': IBr}
        contrast_pairs_f = {'CA': CAf,'CB': CBf,'IA': IAf,'IB': IBf}

        return cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2