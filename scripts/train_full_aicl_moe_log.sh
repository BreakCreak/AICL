#!/bin/bash

# 设置默认GPU（如果没有指定）
GPU_ID=${1:-0}

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "使用 GPU $GPU_ID 开始训练 (d) Full AICL-MoE 模型..."

# 训练完整AICL-MoE模型 (d) 并记录日志
nohup python main_thumos.py \
--exp_name Thumos_FullAICLMoE \
--model_name FullAICLMoE \
--model_variant full \
--num_epochs 900 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_full_aicl_moe.log 2>&1 &

FULL_PID=$!
echo "Full AICL-MoE 模型训练已启动，GPU: $GPU_ID，PID: $FULL_PID"
echo "$FULL_PID" > logs/full_aicl_moe.pid