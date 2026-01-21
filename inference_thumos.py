import os
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils import misc_utils
from run_eval import evaluate
from config.config_thumos import class_dict

def load_weight(model_file, net):
    if model_file is not None:
        print("Loading model weights from:", model_file)
        net.load_state_dict(torch.load(model_file), strict=False)

def inference(net, config, test_loader, model_file=None):
    np.set_printoptions(formatter={'float_kind': "{:.4f}".format})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    # 加载权重
    load_weight(model_file, net)

    final_res = {'version': 'VERSION 1.3', 'results': {},
                 'external_data': {'used': True, 'details': 'Features from I3D Network'}}

    num_correct = 0.
    num_total = 0.

    for _data, _label, temp_anno, vid_name, vid_num_seg in tqdm(test_loader):
        _data = _data.to(device)
        _label = _label.to(device)
        vid = vid_name[0]

        with torch.no_grad():
            # forward
            cas, action_flow, action_rgb, contrast_pairs, contrast_pairs_r, contrast_pairs_f, actionness1, actionness2, aness_bin1, aness_bin2 = net(_data)

            # ====================== 多分支融合 ======================
            # 直接用 soft gate 融合四分支
            emb_rgb = action_rgb.unsqueeze(1).permute(0,2,1)
            emb_flow = action_flow.unsqueeze(1).permute(0,2,1)
            emb_mixed1 = contrast_pairs_r['CA']  # 你可以改成实际 embedding
            emb_mixed2 = contrast_pairs_f['CA']

            # 使用新的 instance_selection_function2 或自定义融合
            combined_cas = misc_utils.instance_selection_function2(
                torch.softmax(cas.detach(), -1),    # CAS 主干
                emb_flow.detach(),
                emb_mixed1.detach(),
                emb_rgb.detach()
            )

            # top-k 取片段
            topk_num = config.num_segments // 8
            _, topk_indices = torch.topk(combined_cas, topk_num, dim=1)
            cas_top = torch.gather(cas, 1, topk_indices)
            cas_top = torch.mean(cas_top, dim=1)
            score_supp = F.softmax(cas_top, dim=1)

            label_np = _label.cpu().numpy()
            score_np = score_supp[0].cpu().numpy()

            # 二值化
            score_np[score_np < config.class_thresh] = 0
            score_np[score_np >= config.class_thresh] = 1
            if np.all(score_np == 0):
                arg = np.argmax(score_supp[0].cpu().data.numpy())
                score_np[arg] = 1

            correct_pred = np.sum(label_np == score_np, axis=1)
            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]

            # action prediction
            pred = np.where(score_np > 0.2)[0]  # 固定阈值
            if len(pred) != 0:
                # 融合四分支 CAS
                cas_pred = combined_cas[0].cpu().numpy()[:, pred]
                cas_pred = np.reshape(cas_pred, (config.num_segments, -1, 1))
                cas_pred = misc_utils.upgrade_resolution(cas_pred, config.scale)

                proposal_dict = {}
                for t in range(len(config.act_thresh)):
                    cas_temp = cas_pred.copy()
                    zero_location = np.where(cas_temp[:,:,0] < config.act_thresh[t])
                    cas_temp[zero_location] = 0
                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(cas_temp[:, c, 0] > 0)
                        seg_list.append(pos)
                    proposals = misc_utils.get_proposal_oic(
                        seg_list, cas_pred.copy(), score_supp[0].cpu().numpy(),
                        pred, config.scale, vid_num_seg[0].cpu().item(),
                        config.feature_fps, config.num_segments, config.gamma
                    )
                    for j in range(len(proposals)):
                        if not proposals[j]:
                            continue
                        class_id = proposals[j][0][0]
                        if class_id not in proposal_dict:
                            proposal_dict[class_id] = []
                        proposal_dict[class_id] += proposals[j]

                # NMS
                final_proposals = []
                for class_id in proposal_dict.keys():
                    final_proposals.append(
                        misc_utils.basnet_nms(proposal_dict[class_id], config.nms_thresh,
                                               config.soft_nms, config.nms_alpha)
                    )
                final_res['results'][vid] = misc_utils.result2json(final_proposals)

    test_acc = num_correct / num_total
    json_path = os.path.join(config.model_path, 'temp_result.json')
    with open(json_path, 'w') as f:
        json.dump(final_res, f)

    mean_ap, _ = evaluate(config.gt_path, json_path, None,
                          tiou_thresholds=np.linspace(0.1, 0.7, 7),
                          plot=False, subset='test', verbose=config.verbose)
    return mean_ap, test_acc
