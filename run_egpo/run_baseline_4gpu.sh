#!/bin/bash
source $(dirname "$0")/utils_env.sh
GPU_IDS="0,1,2,3"

echo ">>> Running EGPO Baseline on 4 GPUs..."
CUDA_VISIBLE_DEVICES=$GPU_IDS python run.py \
    --models "configs/models/deepseek/vllm_deepseek_egpo_baseline.py" "configs/models/qwen3/vllm_qwen3_series.py" \
    --datasets "configs/datasets/egpo_datasets.py" \
    --work-dir "outputs/egpo_baseline_full" \
    --max-num-workers 4 \
    2>&1 | tee "outputs/logs/baseline_4gpu.log"