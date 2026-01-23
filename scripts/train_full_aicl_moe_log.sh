#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练完整AICL-MoE模型 (d) 并记录日志
echo "开始训练 (d) Full AICL-MoE 模型..."
nohup python main_thumos.py \
--exp_name Thumos_FullAICLMoE \
--model_name FullAICLMoE \
--model_variant full \
--num_epochs 900 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_full_aicl_moe.log 2>&1 &

FULL_PID=$!
echo "Full AICL-MoE 模型训练已启动，PID: $FULL_PID"
echo "$FULL_PID" > logs/full_aicl_moe.pid