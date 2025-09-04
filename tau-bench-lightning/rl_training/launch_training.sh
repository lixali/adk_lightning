#!/bin/bash
# Launch PPO training with 8 H100 GPUs

set -e

echo "🚀 Launching PPO Training for Gemma3..."

# Check if setup was run
if [ ! -f "/home/lixali_google_com/data/adk-lightning/tau-bench-lightning/rl_training/.setup_complete" ]; then
    echo "❌ Please run ./rl_training/setup_rl_training.sh first"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export WANDB_MODE=offline
export WANDB_DISABLED=true

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=1

# Create output directories
mkdir -p /home/lixali_google_com/data/adk-lightning/tau-bench-lightning/rl_training/outputs
mkdir -p /home/lixali_google_com/data/adk-lightning/tau-bench-lightning/rl_training/logs

# Set cache directories to use data partition
export HF_HOME="/home/lixali_google_com/data/model_cache/huggingface"
export TRANSFORMERS_CACHE="/home/lixali_google_com/data/model_cache/transformers"
export TORCH_HOME="/home/lixali_google_com/data/model_cache/torch"
export HF_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"

echo "📁 Using cache directories in /home/lixali_google_com/data/model_cache/"

# Check environment
echo "🔍 Environment check:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"

# Launch training with accelerate (using the same arguments as simple_training_test.sh)
cd /home/lixali_google_com/data/adk-lightning/tau-bench-lightning
accelerate launch \
    --config_file rl_training/configs/accelerate_config.yaml \
    --main_process_port 29500 \
    rl_training/ppo_trainer.py \
    --output_dir rl_training/outputs \
    --log_dir rl_training/logs \
    --model_name google/gemma-3-12b-it \
    --num_episodes 50 \
    --rollouts_per_episode 20 \
    --learning_rate 1e-5 \
    --save_steps 5 \
    --logging_steps 1 \
    --gradient_accumulation_steps 4 \
    --max_sequence_length 1024

echo "✅ Training completed!"
