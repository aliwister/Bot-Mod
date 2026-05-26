#!/bin/bash

# Load environment variables from .env file
set -a
source "$(dirname "$0")/.env"
set +a
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# Start vLLM with the specified configuration
exec .venv/bin/vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 3 \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 \
    --port 8000 
