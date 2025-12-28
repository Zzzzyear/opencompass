#!/bin/bash
# 黄金环境配置
export VLLM_USE_V1=1
unset PYTORCH_CUDA_ALLOC_CONF
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONPATH=$(pwd):$PYTHONPATH