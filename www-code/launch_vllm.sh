#!/bin/bash
# Serve the three LLMs the experiments talk to (OpenAI-compatible vLLM servers):
#   BASE     Qwen/Qwen3-8B                      (moderator LLM and one agent backbone)
#   BASE+1   mistralai/Mistral-7B-Instruct-v0.3 (agent backbone; moderator in E19)
#   BASE+2   meta-llama/Llama-3.1-8B-Instruct   (agent backbone; moderator in E19)
# All three fit on one 80 GB GPU. Clients read BOTMOD_PORT_BASE (default 8000).
#   GPU=0 BOTMOD_PORT_BASE=8000 bash launch_vllm.sh
set -u
GPU=${GPU:-0}
BASE=${BOTMOD_PORT_BASE:-8000}
mkdir -p logs
start() {  # port, gpu memory fraction, model, extra args...
  local port=$1 mem=$2 model=$3; shift 3
  if curl -s localhost:$port/v1/models | grep -q "$model"; then echo "$port up"; return; fi
  CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$model" --port $port --max-model-len 8192 \
    --gpu-memory-utilization $mem "$@" > logs/vllm$port.log 2>&1 &
  for _ in $(seq 1 120); do
    sleep 5; curl -s localhost:$port/v1/models | grep -q "$model" && { echo "$port up"; return; }
  done
  echo "$port FAILED (see logs/vllm$port.log)"
}
# started one at a time: engines profiling memory concurrently on one GPU leave no room for KV cache
start $BASE       0.34 Qwen/Qwen3-8B --reasoning-parser qwen3
start $((BASE+1)) 0.30 mistralai/Mistral-7B-Instruct-v0.3
start $((BASE+2)) 0.30 meta-llama/Llama-3.1-8B-Instruct
