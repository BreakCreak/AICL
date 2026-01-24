#!/bin/bash

# 创建日志目录
mkdir -p logs

echo "==================================="
echo "开始优化版并行运行所有消融实验 (双GPU)"
echo "时间: $(date)"
echo "==================================="

# 分配任务到两个GPU
# GPU 0: (a) Single-Expert 和 (b) Multi-Expert (No Mixed)
# GPU 1: (c) Multi-Expert + Mixed (No Gate) 和 (d) Full AICL-MoE

# 启动 GPU 0 上的任务
echo "启动 GPU 0 上的任务..."

# 启动 (a) Single-Expert 模型 (GPU 0)
echo "启动 (a) Single-Expert 模型训练 (GPU 0)..."
nohup bash scripts/train_single_expert_log.sh 0 > logs/single_expert_master.log 2>&1 &
SINGLE_PID=$!

echo "Single Expert PID: $SINGLE_PID (saved to logs/single_expert.pid)"

# 启动 (b) Multi-Expert (No Mixed) 模型 (GPU 0)
sleep 2  # 稍微延迟以避免同时启动冲突
echo "启动 (b) Multi-Expert (No Mixed) 模型训练 (GPU 0)..."
nohup bash scripts/train_multi_no_mixed_log.sh 0 > logs/multi_no_mixed_master.log 2>&1 &
MIXED_PID=$!

echo "Multi No Mixed PID: $MIXED_PID (saved to logs/multi_no_mixed.pid)"

# 启动 GPU 1 上的任务
echo "启动 GPU 1 上的任务..."

# 启动 (c) Multi-Expert + Mixed (No Gate) 模型 (GPU 1)
sleep 2  # 稍微延迟以避免同时启动冲突
echo "启动 (c) Multi-Expert + Mixed (No Gate) 模型训练 (GPU 1)..."
nohup bash scripts/train_multi_mixed_no_gate_log.sh 1 > logs/multi_mixed_no_gate_master.log 2>&1 &
NO_GATE_PID=$!

echo "Multi Mixed No Gate PID: $NO_GATE_PID (saved to logs/multi_mixed_no_gate.pid)"

# 启动 (d) Full AICL-MoE 模型 (GPU 1)
sleep 2  # 稍微延迟以避免同时启动冲突
echo "启动 (d) Full AICL-MoE 模型训练 (GPU 1)..."
nohup bash scripts/train_full_aicl_moe_log.sh 1 > logs/full_aicl_moe_master.log 2>&1 &
FULL_PID=$!

echo "Full AICL-MoE PID: $FULL_PID (saved to logs/full_aicl_moe.pid)"

echo ""
echo "==================================="
echo "所有消融实验已启动（双GPU优化版）！"
echo "时间: $(date)"
echo ""
echo "进程信息:"
echo "  GPU 0:"
echo "    (a) Single-Expert PID: $SINGLE_PID"
echo "    (b) Multi-Expert No Mixed PID: $MIXED_PID" 
echo "  GPU 1:"
echo "    (c) Multi-Expert + Mixed No Gate PID: $NO_GATE_PID"
echo "    (d) Full AICL-MoE PID: $FULL_PID"
echo ""
echo "日志文件位置:"
echo "  logs/train_single_expert.log"
echo "  logs/train_multi_no_mixed.log" 
echo "  logs/train_multi_mixed_no_gate.log"
echo "  logs/train_full_aicl_moe.log"
echo "  logs/*_master.log (主控制日志)"
echo ""
echo "要监控GPU使用情况，可使用:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "要监控训练进度，可使用以下命令:"
echo "  tail -f logs/train_full_aicl_moe.log"
echo "  tail -f logs/train_single_expert.log"
echo "  tail -f logs/train_multi_no_mixed.log"
echo "  tail -f logs/train_multi_mixed_no_gate.log"
echo ""
echo "要检查所有进程状态，可使用:"
echo "  bash scripts/check_all_status.sh"
echo "==================================="