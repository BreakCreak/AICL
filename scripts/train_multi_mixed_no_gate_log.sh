#!/bin/bash

# 设置默认GPU（如果没有指定）
GPU_ID=${1:-0}

# 创建日志目录（如果不存在）
mkdir -p logs

# 设置环境变量
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "使用 GPU $GPU_ID 开始训练 (c) Multi-Expert + Mixed (No Gate) 模型..."

# 训练多专家混合无门控模型 (c) 并记录日志
nohup python main_thumos.py \
--exp_name Thumos_MultiMixedNoGate \
--model_name MultiExpertMixedNoGate \
--model_variant multi_mixed_no_gate \
--num_epochs 900 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train_multi_mixed_no_gate.log 2>&1 &

NO_GATE_PID=$!
echo "Multi Expert Mixed No Gate 模型训练已启动，GPU: $GPU_ID，PID: $NO_GATE_PID"
echo "$NO_GATE_PID" > logs/multi_mixed_no_gate.pid