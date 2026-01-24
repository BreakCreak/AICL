#!/bin/bash

# 设置默认GPU（如果没有指定）
GPU_ID=${1:-0}

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "使用 GPU $GPU_ID 开始训练 (a) Single-Expert 模型..."

# 训练单一专家模型 (a) 并记录日志
nohup python main_thumos.py \
--exp_name Thumos_SingleExpert \
--model_name SingleExpert \
--model_variant single \
--num_epochs 900 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_single_expert.log 2>&1 &

SINGLE_PID=$!
echo "Single Expert 模型训练已启动，GPU: $GPU_ID，PID: $SINGLE_PID"
echo "$SINGLE_PID" > logs/single_expert.pid