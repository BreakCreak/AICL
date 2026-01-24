#!/bin/bash

# 默认GPU分配
GPU_A=0  # (a) Single-Expert
GPU_B=0  # (b) Multi-Expert (No Mixed)
GPU_C=1  # (c) Multi-Expert + Mixed (No Gate) 
GPU_D=1  # (d) Full AICL-MoE

# 允许用户通过参数自定义GPU分配
if [ $# -eq 4 ]; then
    GPU_A=$1
    GPU_B=$2
    GPU_C=$3
    GPU_D=$4
fi

# 创建日志目录
mkdir -p logs

echo "==================================="
echo "自定义GPU分配运行所有消融实验"
echo "当前分配: A->$GPU_A, B->$GPU_B, C->$GPU_C, D->$GPU_D"
echo "时间: $(date)"
echo "==================================="

# 启动 (a) Single-Expert 模型
echo "启动 (a) Single-Expert 模型训练 (GPU $GPU_A)..."
nohup bash scripts/train_single_expert_log.sh $GPU_A > logs/single_expert_master.log 2>&1 &
SINGLE_PID=$!
echo "Single Expert PID: $SINGLE_PID (saved to logs/single_expert.pid)"

# 启动 (b) Multi-Expert (No Mixed) 模型
sleep 2  # 稍微延迟以避免同时启动冲突
echo "启动 (b) Multi-Expert (No Mixed) 模型训练 (GPU $GPU_B)..."
nohup bash scripts/train_multi_no_mixed_log.sh $GPU_B > logs/multi_no_mixed_master.log 2>&1 &
MIXED_PID=$!
echo "Multi No Mixed PID: $MIXED_PID (saved to logs/multi_no_mixed.pid)"

# 启动 (c) Multi-Expert + Mixed (No Gate) 模型
sleep 2  # 稍微延迟以避免同时启动冲突
echo "启动 (c) Multi-Expert + Mixed (No Gate) 模型训练 (GPU $GPU_C)..."
nohup bash scripts/train_multi_mixed_no_gate_log.sh $GPU_C > logs/multi_mixed_no_gate_master.log 2>&1 &
NO_GATE_PID=$!
echo "Multi Mixed No Gate PID: $NO_GATE_PID (saved to logs/multi_mixed_no_gate.pid)"

# 启动 (d) Full AICL-MoE 模型
sleep 2  # 稍微延迟以避免同时启动冲突
echo "启动 (d) Full AICL-MoE 模型训练 (GPU $GPU_D)..."
nohup bash scripts/train_full_aicl_moe_log.sh $GPU_D > logs/full_aicl_moe_master.log 2>&1 &
FULL_PID=$!
echo "Full AICL-MoE PID: $FULL_PID (saved to logs/full_aicl_moe.pid)"

echo ""
echo "==================================="
echo "所有消融实验已启动（自定义GPU分配）！"
echo "时间: $(date)"
echo ""
echo "GPU分配详情:"
echo "  GPU $GPU_A: (a) Single-Expert PID: $SINGLE_PID"
echo "  GPU $GPU_B: (b) Multi-Expert No Mixed PID: $MIXED_PID" 
echo "  GPU $GPU_C: (c) Multi-Expert + Mixed No Gate PID: $NO_GATE_PID"
echo "  GPU $GPU_D: (d) Full AICL-MoE PID: $FULL_PID"
echo ""
echo "使用方法示例:"
echo "  默认分配 (0,0,1,1): bash scripts/run_all_ablations_custom_gpu.sh"
echo "  全部使用GPU 0: bash scripts/run_all_ablations_custom_gpu.sh 0 0 0 0"
echo "  均匀分配 (0,1,0,1): bash scripts/run_all_ablations_custom_gpu.sh 0 1 0 1"
echo "  自定义分配: bash scripts/run_all_ablations_custom_gpu.sh [GPU_A] [GPU_B] [GPU_C] [GPU_D]"
echo ""
echo "日志文件位置:"
echo "  logs/train_single_expert.log"
echo "  logs/train_multi_no_mixed.log" 
echo "  logs/train_multi_mixed_no_gate.log"
echo "  logs/train_full_aicl_moe.log"
echo ""
echo "要监控GPU使用情况，可使用: watch -n 1 nvidia-smi"
echo "要监控训练进度，可使用: tail -f logs/train_*.log"
echo "==================================="