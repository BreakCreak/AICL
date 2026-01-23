#!/bin/bash

# 创建日志目录
mkdir -p logs

echo "==================================="
echo "开始一键运行所有消融实验"
echo "时间: $(date)"
echo "==================================="

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 启动 (a) Single-Expert 模型
echo "启动 (a) Single-Expert 模型训练..."
nohup bash scripts/train_single_expert_log.sh > logs/single_expert_master.log 2>&1 &
SINGLE_PID=$!

echo "Single Expert PID: $SINGLE_PID (saved to logs/single_expert.pid)"

# 启动 (b) Multi-Expert (No Mixed) 模型
sleep 3  # 稍微延迟以避免同时启动冲突
echo "启动 (b) Multi-Expert (No Mixed) 模型训练..."
nohup bash scripts/train_multi_no_mixed_log.sh > logs/multi_no_mixed_master.log 2>&1 &
MIXED_PID=$!

echo "Multi No Mixed PID: $MIXED_PID (saved to logs/multi_no_mixed.pid)"

# 启动 (c) Multi-Expert + Mixed (No Gate) 模型
sleep 3  # 稍微延迟以避免同时启动冲突
echo "启动 (c) Multi-Expert + Mixed (No Gate) 模型训练..."
nohup bash scripts/train_multi_mixed_no_gate_log.sh > logs/multi_mixed_no_gate_master.log 2>&1 &
NO_GATE_PID=$!

echo "Multi Mixed No Gate PID: $NO_GATE_PID (saved to logs/multi_mixed_no_gate.pid)"

# 启动 (d) Full AICL-MoE 模型
sleep 3  # 稍微延迟以避免同时启动冲突
echo "启动 (d) Full AICL-MoE 模型训练..."
nohup bash scripts/train_full_aicl_moe_log.sh > logs/full_aicl_moe_master.log 2>&1 &
FULL_PID=$!

echo "Full AICL-MoE PID: $FULL_PID (saved to logs/full_aicl_moe.pid)"

echo ""
echo "==================================="
echo "所有消融实验已启动！"
echo "时间: $(date)"
echo ""
echo "进程信息:"
echo "  (a) Single-Expert PID: $SINGLE_PID"
echo "  (b) Multi-Expert No Mixed PID: $MIXED_PID" 
echo "  (c) Multi-Expert + Mixed No Gate PID: $NO_GATE_PID"
echo "  (d) Full AICL-MoE PID: $FULL_PID"
echo ""
echo "日志文件位置:"
echo "  logs/train_single_expert.log"
echo "  logs/train_multi_no_mixed.log" 
echo "  logs/train_multi_mixed_no_gate.log"
echo "  logs/train_full_aicl_moe.log"
echo "  logs/*_master.log (主控制日志)"
echo ""
echo "要监控训练进度，可使用以下命令:"
echo "  tail -f logs/train_full_aicl_moe.log"
echo "  tail -f logs/train_single_expert.log"
echo "  tail -f logs/train_multi_no_mixed.log"
echo "  tail -f logs/train_multi_mixed_no_gate.log"
echo ""
echo "要检查所有进程状态，可使用:"
echo "  ps aux | grep python | grep -v grep"
echo "==================================="