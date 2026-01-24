#!/bin/bash

# 设置默认GPU（如果没有指定）
GPU_ID=${1:-0}

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "使用 GPU $GPU_ID 开始训练 (b) Multi-Expert (No Mixed) 模型..."

# 训练多专家无混合模型 (b) 并记录日志
nohup python main_thumos.py \
--exp_name Thumos_MultiNoMixed \
--model_name MultiExpertNoMixed \
--model_variant multi_no_mixed \
--num_epochs 900 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_multi_no_mixed.log 2>&1 &

MIXED_PID=$!
echo "Multi Expert No Mixed 模型训练已启动，GPU: $GPU_ID，PID: $MIXED_PID"
echo "$MIXED_PID" > logs/multi_no_mixed.pid