#!/bin/bash
source $(dirname "$0")/utils_env.sh

LOG_DIR="outputs/logs/test_$(date +%Y%m%d)"
mkdir -p $LOG_DIR

# 仅测试 DeepSeek-Math-Instruct
echo ">>> Running Smoke Test on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python run.py \
    --models "configs/models/deepseek/vllm_deepseek_egpo_baseline.py" \
    --datasets "configs/datasets/egpo_datasets.py" \
    --work-dir "outputs/egpo_smoke_test" \
    -f "deepseek-math-7b-instruct-vllm" \
    --debug \
    -d 2 2>&1 | tee "$LOG_DIR/smoke_test.log"