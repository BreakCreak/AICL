
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=1

echo "使用 GPU $GPU_ID 开始训练 (d) Full AICL-MoE 模型..."
nohup python main_thumos.py \
--exp_name Thumos_2 \
--model_name ThumosModel \
--num_epochs 400 \
--detection_inf_step 50 \
--soft_nms \
--data_path '../THUMOS14' > logs/train.log 2>&1 &