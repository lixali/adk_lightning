#!/bin/bash
# Simple training launcher without accelerate (for testing)

set -e

echo "🚀 Starting Simple PPO Training Test"

# Check if setup was run
if [ ! -f "/home/lixali_google_com/data/adk-lightning/tau-bench-lightning/rl_training/.setup_complete" ]; then
    echo "❌ Please run ./rl_training/setup_rl_training.sh first"
    exit 1
fi

# Check Google Cloud authentication (skip for now)
echo "ℹ️  Skipping Google Cloud authentication check for simple test..."

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Create output directories
mkdir -p /home/lixali_google_com/data/adk-lightning/tau-bench-lightning/rl_training/outputs
mkdir -p /home/lixali_google_com/data/adk-lightning/tau-bench-lightning/rl_training/logs

echo "🎯 Starting simple training test..."

# Set cache directories to use data partition
export HF_HOME="/home/lixali_google_com/data/model_cache/huggingface"
export TRANSFORMERS_CACHE="/home/lixali_google_com/data/model_cache/transformers"
export TORCH_HOME="/home/lixali_google_com/data/model_cache/torch"
export HF_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"

echo "📁 Using cache directories in /home/lixali_google_com/data/model_cache/"

# Run training directly with Python (single GPU for testing)
cd /home/lixali_google_com/data/adk-lightning/tau-bench-lightning
CUDA_VISIBLE_DEVICES=0,1,2,3 python rl_training/ppo_trainer.py \
    --output_dir rl_training/outputs \
    --log_dir rl_training/logs \
    --model_name google/gemma-3-12b-it \
    --num_episodes 2 \
    --rollouts_per_episode 5 \
    --learning_rate 1e-5 \
    --save_steps 10 \
    --logging_steps 1 \
    --test_run

echo "✅ Simple training test completed!"
