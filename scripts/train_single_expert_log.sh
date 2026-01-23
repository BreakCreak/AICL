#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练单一专家模型 (a) 并记录日志
echo "开始训练 (a) Single-Expert 模型..."
nohup python main_thumos.py \
--exp_name Thumos_SingleExpert \
--model_name SingleExpert \
--model_variant single \
--num_epochs 400 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_single_expert.log 2>&1 &

SINGLE_PID=$!
echo "Single Expert 模型训练已启动，PID: $SINGLE_PID"
echo "$SINGLE_PID" > logs/single_expert.pid