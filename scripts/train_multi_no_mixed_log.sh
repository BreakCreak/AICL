#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练多专家无混合模型 (b) 并记录日志
echo "开始训练 (b) Multi-Expert (No Mixed) 模型..."
nohup python main_thumos.py \
--exp_name Thumos_MultiNoMixed \
--model_name MultiExpertNoMixed \
--model_variant multi_no_mixed \
--num_epochs 400 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_multi_no_mixed.log 2>&1 &

MIXED_PID=$!
echo "Multi Expert No Mixed 模型训练已启动，PID: $MIXED_PID"
echo "$MIXED_PID" > logs/multi_no_mixed.pid