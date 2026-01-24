#!/bin/bash

echo "==================================="
echo "检查所有消融实验状态"
echo "时间: $(date)"
echo "==================================="

# 检查日志目录
if [ ! -d "logs" ]; then
    echo "错误: logs 目录不存在"
    exit 1
fi

# 检查各个PID文件并验证进程是否存在
echo "检查各模型训练进程状态:"

if [ -f "logs/single_expert.pid" ]; then
    SINGLE_PID=$(cat logs/single_expert.pid)
    if ps -p $SINGLE_PID > /dev/null 2>&1; then
        echo "✓ (a) Single-Expert 进程运行中 (PID: $SINGLE_PID)"
    else
        echo "✗ (a) Single-Expert 进程未运行 (PID: $SINGLE_PID)"
    fi
else
    echo "? (a) Single-Expert PID文件不存在"
fi

if [ -f "logs/multi_no_mixed.pid" ]; then
    MIXED_PID=$(cat logs/multi_no_mixed.pid)
    if ps -p $MIXED_PID > /dev/null 2>&1; then
        echo "✓ (b) Multi-Expert No Mixed 进程运行中 (PID: $MIXED_PID)"
    else
        echo "✗ (b) Multi-Expert No Mixed 进程未运行 (PID: $MIXED_PID)"
    fi
else
    echo "? (b) Multi-Expert No Mixed PID文件不存在"
fi

if [ -f "logs/multi_mixed_no_gate.pid" ]; then
    NO_GATE_PID=$(cat logs/multi_mixed_no_gate.pid)
    if ps -p $NO_GATE_PID > /dev/null 2>&1; then
        echo "✓ (c) Multi-Expert + Mixed No Gate 进程运行中 (PID: $NO_GATE_PID)"
    else
        echo "✗ (c) Multi-Expert + Mixed No Gate 进程未运行 (PID: $NO_GATE_PID)"
    fi
else
    echo "? (c) Multi-Expert + Mixed No Gate PID文件不存在"
fi

if [ -f "logs/full_aicl_moe.pid" ]; then
    FULL_PID=$(cat logs/full_aicl_moe.pid)
    if ps -p $FULL_PID > /dev/null 2>&1; then
        echo "✓ (d) Full AICL-MoE 进程运行中 (PID: $FULL_PID)"
    else
        echo "✗ (d) Full AICL-MoE 进程未运行 (PID: $FULL_PID)"
    fi
else
    echo "? (d) Full AICL-MoE PID文件不存在"
fi

echo ""
echo "当前 Python 进程:"
ps aux | grep python | grep -v grep | grep -v check_all_status

echo ""
echo "GPU 使用情况:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi 未找到或无GPU"

echo ""
echo "日志文件大小:"
for log in logs/train_*.log; do
    if [ -f "$log" ]; then
        size=$(du -h "$log" 2>/dev/null | cut -f1)
        lines=$(wc -l < "$log")
        echo "- $(basename $log): ${size:-0KB} (${lines} 行)"
    fi
done

echo ""
echo "==================================="
echo "检查完成"
echo "==================================="