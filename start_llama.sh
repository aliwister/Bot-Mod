#!/bin/bash

# Load environment variables from .env file
set -a
source "$(dirname "$0")/.env"
set +a

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=7

# Start vLLM with the specified configuration
exec .venv/bin/vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 \
  --port 8002
