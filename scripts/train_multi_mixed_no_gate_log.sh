#!/bin/bash

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 训练多专家混合无门控模型 (c) 并记录日志
echo "开始训练 (c) Multi-Expert + Mixed (No Gate) 模型..."
nohup python main_thumos.py \
--exp_name Thumos_MultiMixedNoGate \
--model_name MultiExpertMixedNoGate \
--model_variant multi_mixed_no_gate \
--num_epochs 400 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_multi_mixed_no_gate.log 2>&1 &

NO_GATE_PID=$!
echo "Multi Expert Mixed No Gate 模型训练已启动，PID: $NO_GATE_PID"
echo "$NO_GATE_PID" > logs/multi_mixed_no_gate.pid