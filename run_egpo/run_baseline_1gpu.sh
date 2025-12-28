#!/bin/bash
# Usage: bash run_baseline_1gpu.sh <GPU_ID>
source $(dirname "$0")/utils_env.sh
GPU_ID=${1:-0}


echo ">>> Running EGPO Baseline on GPU $GPU_ID..."
CUDA_VISIBLE_DEVICES=$GPU_ID python run.py \
    --models "configs/models/deepseek/vllm_deepseek_egpo_baseline.py" "configs/models/qwen3/vllm_qwen3_series.py" \
    --datasets "configs/datasets/egpo_datasets.py" \
    --work-dir "outputs/egpo_baseline_full" \
    --max-num-workers 1 \
    2>&1 | tee "outputs/logs/baseline_gpu${GPU_ID}.log"