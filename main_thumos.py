import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F

from inference_thumos import inference
from utils import misc_utils
from torch.utils.data import Dataset
from dataset.thumos_features import ThumosFeature
from utils.loss import CrossEntropyLoss, GeneralizedCE, GateConstraintLoss, EntropyRegularizationLoss
from config.config_thumos import Config, parse_args, class_dict
import importlib
from models.model import AICL
from models.ablation_model import AICLAblation

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})

np.set_printoptions(threshold=np.inf)

def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "CAS_Only.pkl")
        print("loading from file for training: ", model_file)
        pretrained_params = torch.load(model_file)

        selected_params = OrderedDict()
        for k, v in pretrained_params.items():
            if 'base_module' in k:
                selected_params[k] = v

        model_dict = net.state_dict()
        model_dict.update(selected_params)
        net.load_state_dict(model_dict)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.1):                #　　0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        tau = 0.07  # 添加temperature参数
        logits /= tau  # 应用temperature缩放
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['IA'], 1),
            torch.mean(contrast_pairs['CA'], 1),
            contrast_pairs['CB']
        )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['IB'], 1),
            torch.mean(contrast_pairs['CB'], 1),
            contrast_pairs['CA']
        )

        CA_refinement = self.NCE(
            torch.mean(contrast_pairs['CA'], 1),
            torch.mean(contrast_pairs['IA'], 1),
            contrast_pairs['CB']
        )

        CB_refinement = self.NCE(
            torch.mean(contrast_pairs['CB'], 1),
            torch.mean(contrast_pairs['IB'], 1),
            contrast_pairs['CA']
        )

        loss = IA_refinement + IB_refinement + CA_refinement + CB_refinement
        return loss


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        # Check if model variant is specified in config
        if hasattr(config, 'model_variant') and config.model_variant:
            self.net = AICLAblation(config, variant=config.model_variant)
        else:
            self.net = AICL(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.net = self.net.to(self.device)

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
        self.criterion = CrossEntropyLoss()
        self.Lgce = GeneralizedCE(q=self.config.q_val)
        # 添加门控约束损失和熵正则化损失
        self.gate_constraint_loss = GateConstraintLoss(weight=0.1)
        self.entropy_reg_loss = EntropyRegularizationLoss(weight=0.01)

        # parameters
        self.best_mAP = -1 # init
        self.step = 0
        self.total_loss_per_epoch = 0


    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "CAS_Only.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc*100, _mean_ap*100))


    def calculate_pesudo_target(self, batch_size, label, topk_indices):
        cls_agnostic_gt = []
        cls_agnostic_neg_gt = []
        for b in range(batch_size):
            label_indices_b = torch.nonzero(label[b, :])[:,0]
            topk_indices_b = topk_indices[b, :, label_indices_b] # topk, num_actions
            cls_agnostic_gt_b = torch.zeros((1, 1, self.config.num_segments)).to(label.device)

            # positive examples
            for gt_i in range(len(label_indices_b)):
                cls_agnostic_gt_b[0, 0, topk_indices_b[:, gt_i]] = 1
            cls_agnostic_gt.append(cls_agnostic_gt_b)

        return torch.cat(cls_agnostic_gt, dim=0)  # B, 1, num_segments


    def calculate_all_losses1(self, contrast_pairs, contrast_pairs_r, contrast_pairs_f, cas_top, label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2, gate_weights, cas_rgb=None, cas_flow=None, cas_mixed1=None, cas_mixed2=None, performance_penalty=None):
        self.contrastive_criterion = ContrastiveLoss()
        loss_contrastive = self.contrastive_criterion(contrast_pairs) + self.contrastive_criterion(contrast_pairs_r) + self.contrastive_criterion(contrast_pairs_f)

        base_loss = self.criterion(cas_top, label)
        class_agnostic_loss = self.Lgce(action_flow.squeeze(1), cls_agnostic_gt.squeeze(1)) + self.Lgce(action_rgb.squeeze(1), cls_agnostic_gt.squeeze(1))

        modality_consistent_loss = 0.5 * F.mse_loss(action_flow, action_rgb) + 0.5 * F.mse_loss(action_rgb, action_flow)
        action_consistent_loss = 0.1 * F.mse_loss(actionness1, actionness2) + 0.1 * F.mse_loss(actionness2, actionness1)

        # 门控约束损失 - 如果提供了分支CAS输出，则计算门控约束损失
        gate_constraint_loss = 0
        if cas_rgb is not None and cas_flow is not None and cas_mixed1 is not None and cas_mixed2 is not None:
            cas_branches = [cas_rgb, cas_flow, cas_mixed1, cas_mixed2]
            # 重新计算cas_top用于门控约束，因为它需要完整的CAS而不是top-k后的
            cas_full = torch.softmax(contrast_pairs['CA'].mean(dim=1, keepdim=True) if 'CA' in contrast_pairs else cas_top.unsqueeze(1), dim=-1) if cas_top.dim() == 2 else torch.softmax(cas_top, dim=-1)
            gate_constraint_loss = self.gate_constraint_loss(cas_top.unsqueeze(1) if cas_top.dim() == 2 else cas_top, cas_branches)
        
        # 门控权重熵正则化损失
        entropy_reg_loss = self.entropy_reg_loss(gate_weights)

        # 提高高IoU阈值下的性能，增加对比损失权重和模态一致性损失
        cost = base_loss + class_agnostic_loss + 5 * modality_consistent_loss + 0.01 * loss_contrastive + 0.1 * action_consistent_loss + gate_constraint_loss + entropy_reg_loss

        return cost

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                mean_ap, test_acc = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))

            print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  best_map={:5.2f}".format(
                    epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0


    def forward_pass(self, _data):
        # 检查模型返回值数量
        result = self.net(_data)
        if len(result) == 11:  # 原来的返回值数量
            cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights = result
            # 设置默认值
            cas_rgb = cas_flow = cas_mixed1 = cas_mixed2 = performance_penalty = None
        else:  # 新的返回值数量，包含分支CAS和性能惩罚
            cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, performance_penalty = result
            # 当前模型返回值中不包含各分支的CAS，但我们可以从performance_penalty推断出来
            cas_rgb = cas_flow = cas_mixed1 = cas_mixed2 = None

        # 增加一个维度以避免 permute 错误
        action_flow = action_flow.unsqueeze(1)  # 将 [B, T] 转换为 [B, 1, T]
        action_rgb = action_rgb.unsqueeze(1)    # 同上

        # 使用新的 instance_selection_function2 来融合更多分支
        combined_cas = misc_utils.instance_selection_function2(torch.softmax(cas.detach(), -1),
                                                           action_flow.permute(0, 2, 1).detach(),
                                                           action_flow.permute(0, 2, 1),
                                                           action_rgb.permute(0, 2, 1))

        _, topk_indices = torch.topk(combined_cas, self.config.num_segments // 8, dim=1)
        cas_top = torch.mean(torch.gather(cas, 1, topk_indices), dim=1)

        return cas_top, topk_indices, action_flow.squeeze(1), action_rgb.squeeze(1), contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, cas_rgb, cas_flow, cas_mixed1, cas_mixed2, performance_penalty


    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):
            # 动态一致性阈值
            consistency_thresh = min(0.9, 0.5 + epoch * 0.01)

            for _data, _label, temp_anno, _, _ in self.train_loader:

                batch_size = _data.shape[0]
                _data, _label = _data.to(self.device), _label.to(self.device)
                self.optimizer.zero_grad()

                # forward pass
                cas_top, topk_indices, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2, gate_weights, cas_rgb, cas_flow, cas_mixed1, cas_mixed2, performance_penalty = self.forward_pass(_data)

                # calcualte pseudo target
                cls_agnostic_gt = self.calculate_pesudo_target(batch_size, _label, topk_indices)

                # losses
                cost = self.calculate_all_losses1(contrast_pairs, contrast_pairs_r, contrast_pairs_f, cas_top, _label, action_flow, action_rgb, cls_agnostic_gt, actionness1, actionness2, gate_weights, cas_rgb, cas_flow, cas_mixed1, cas_mixed2, performance_penalty)

                cost.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += cost.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)



def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
