import os
import sys
import math
import glob
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.interpolate import interp1d
from tqdm import tqdm


def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, lambda_=0.25, gamma=0.2, expand_ratio=0.1):
    t_factor = (16 * v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):   # 如果该视频只判别出一种动作len(tList)=1, 两种就是2
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)

            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])  # 求该视频中第j个proposal中所有帧得分的均值

                len_proposal = len(grouped_temp_list[j])

                outer_s = max(0, int(grouped_temp_list[j][0] - lambda_ * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + lambda_ * len_proposal))   # wtcam.shape[0]=18000

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(
                    range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))

                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                
                # 边界扩张 - 修改此部分以放宽边界
                start = grouped_temp_list[j][0] * t_factor
                end = (grouped_temp_list[j][-1] + 1) * t_factor
                length = end - start
                # 扩张比例从0.1增加到0.2或0.3
                expansion_amount = expand_ratio * length
                start = max(0, start - expansion_amount)
                end = min(v_len * 16 / sampling_frames, end + expansion_amount)  # 确保不超过视频长度
                
                c_temp.append([c_pred[i], c_score, start, end])    # 动作类别，置信度，开始时刻，结束时刻
        temp.append(c_temp)
    return temp


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def basnet_nms(proposals, thresh, soft_nms=False, nms_alpha=0):
    # 分段NMS：对短proposal和长proposal使用不同的IOU阈值
    proposals = np.array(proposals)

    x1 = proposals[:, 2]        # start
    x2 = proposals[:, 3]        # end
    scores = proposals[:, 1]

    areas = x2 - x1  # 修改这里，不需要+1，因为是时间段
    order = scores.argsort()[::-1]    #　[::-1] 顺序倒置

    keep = []
    not_keep = []
    while order.size > 0:
        i = order[0]
        
        # 计算提案长度（以帧为单位）
        proposal_length = x2[i] - x1[i]
        
        # 根据提案长度选择不同的NMS IOU阈值 - 实现分段NMS
        if proposal_length < 20:  # 短proposal
            current_thresh = 0.8
        else:  # 长proposal
            current_thresh = 0.6
        
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        if soft_nms:
            inv_inds = np.where(iou >= current_thresh)[0]
            props_mod = proposals[order[inv_inds + 1]]

            for k in range(props_mod.shape[0]):
                props_mod[k, 1] = props_mod[k, 1] * np.exp(-np.square(iou[inv_inds][k]) / nms_alpha)

            not_keep.extend(props_mod.tolist())

        inds = np.where(iou < current_thresh)[0]
        order = order[inds + 1]

    if soft_nms:
        keep.extend(not_keep)
    # print(np.array(keep).shape)
    # exit()
    return keep


def instance_selection_function(cas, *actionness):
    return (cas + sum(actionness)) / (1 + len(actionness))


def instance_selection_function2(cas_r, cas_f, cas_flow, cas_rgb):
    # 进一步优化的融合策略，特别关注高IoU阈值下的性能
    # 增强对动作边界精确定位的支持
    combine_cas = (0.35*cas_r + 0.35*cas_f + 0.65*cas_flow + 0.65*cas_rgb) / 2.0
    return combine_cas


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def result2json(result):
    from config.config_thumos import class_dict
    result_file = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            line = {'label': class_dict[result[i][j][0]], 'score300': result[i][j][1],   # 300
                    'segment': [result[i][j][2], result[i][j][3]]}
            result_file.append(line)
    return result_file
