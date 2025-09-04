#!/bin/bash
# Setup script for RL training environment with 8 H100 GPUs

set -e

echo "🚀 Setting up RL Training Environment for Gemma3 PPO Training"

# Check for CUDA availability
echo "🔍 Checking CUDA availability..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ CUDA not found. Please install CUDA drivers."
    exit 1
fi

nvidia-smi
echo "✅ CUDA detected"

# Check number of GPUs
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "🎯 Detected $GPU_COUNT GPUs"

if [ $GPU_COUNT -lt 8 ]; then
    echo "⚠️  Warning: Only $GPU_COUNT GPUs detected. This setup is optimized for 8 H100 GPUs."
    echo "   Training will still work but may be slower."
fi

# Check current conda environment
echo "🐍 Using current conda environment: $CONDA_DEFAULT_ENV"
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected. Using system Python."
    echo "   It's recommended to activate a conda environment first."
fi

# Upgrade pip in current environment
echo "⬆️  Upgrading pip in current environment..."
pip install --upgrade pip

# Install PyTorch with CUDA support first
echo "🔥 Installing PyTorch with CUDA support..."
# Try conda first, fallback to pip
if command -v conda &> /dev/null; then
    echo "   Using conda for PyTorch installation..."
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia || {
        echo "   Conda installation failed, falling back to pip..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    }
else
    echo "   Using pip for PyTorch installation..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install some dependencies via conda if available
if command -v conda &> /dev/null; then
    echo "📦 Installing conda packages..."
    conda install -y numpy scipy ninja packaging -c conda-forge || echo "⚠️  Some conda packages failed to install"
fi

# Install requirements
echo "📦 Installing pip requirements..."
pip install -r rl_training/requirements.txt

# Install flash attention if available
echo "⚡ Installing Flash Attention..."
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention installation failed (optional)"

# Verify installations
echo "🧪 Verifying installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Setup Google Cloud credentials
echo "☁️  Setting up Google Cloud credentials..."
if [ -z "$GOOGLE_CLOUD_PROJECT" ]; then
    echo "⚠️  Warning: GOOGLE_CLOUD_PROJECT environment variable not set."
    echo "   Please set it before running training:"
    echo "   export GOOGLE_CLOUD_PROJECT=your-project-id"
fi

if [ ! -f "$HOME/.config/gcloud/application_default_credentials.json" ]; then
    echo "🔑 Please authenticate with Google Cloud:"
    echo "   gcloud auth application-default login"
fi

# Create output directories
echo "📁 Creating output directories..."
mkdir -p rl_training/outputs
mkdir -p rl_training/checkpoints
mkdir -p rl_training/logs

# Download base model (optional)
echo "🤖 Checking for base model..."
python -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b-it')
    print('✅ Base model accessible')
except:
    print('⚠️  Base model not accessible. Will download during training.')
"

# Create configuration files
echo "⚙️  Creating configuration files..."

# Create training configuration
cat > rl_training/configs/ppo_config.yaml << 'EOF'
# PPO Training Configuration for Gemma3
model:
  name: "google/gemma-2-9b-it"  # Change to gemma3-12b when available
  use_flash_attention: true
  
training:
  num_episodes: 50
  rollouts_per_episode: 100
  learning_rate: 1.0e-5
  critic_learning_rate: 1.0e-4
  ppo_epochs: 4
  clip_epsilon: 0.2
  kl_coeff: 0.02
  max_kl: 0.1
  gamma: 0.99
  gae_lambda: 0.95
  max_grad_norm: 1.0
  temperature: 0.7
  
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  
hardware:
  gpu_count: 8
  tensor_parallel_size: 8
  gpu_memory_utilization: 0.9
  
output:
  output_dir: "./rl_training/outputs"
  save_every: 5
  
wandb:
  project: "gemma3-tau-bench-ppo"
  enabled: true
  
data:
  focus_env: "retail"
  focus_on_wrong_decisions: true
  
reward_model:
  provider: "vertex_ai"
  model: "gemini-2.5-pro"
EOF

# Create launch script
cat > rl_training/launch_training.sh << 'EOF'
#!/bin/bash
# Launch PPO training with 8 H100 GPUs

echo "🚀 Launching PPO Training for Gemma3..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate virtual environment
source rl_venv/bin/activate

# Check environment
echo "🔍 Environment check:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GOOGLE_CLOUD_PROJECT: $GOOGLE_CLOUD_PROJECT"
python -c "import torch; print(f'Available GPUs: {torch.cuda.device_count()}')"

# Launch training with accelerate
accelerate launch \
    --config_file rl_training/configs/accelerate_config.yaml \
    --main_process_port 29500 \
    rl_training/ppo_trainer.py \
    --config rl_training/configs/ppo_config.yaml

echo "✅ Training completed!"
EOF

chmod +x rl_training/launch_training.sh

# Create accelerate config
mkdir -p rl_training/configs
cat > rl_training/configs/accelerate_config.yaml << 'EOF'
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# Create monitoring script
cat > rl_training/monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""Monitor training progress and GPU utilization"""

import time
import psutil
import GPUtil
import json
import os
from datetime import datetime

def monitor_resources():
    """Monitor system resources during training"""
    
    while True:
        timestamp = datetime.now().isoformat()
        
        # GPU utilization
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil * 100,
                'temperature': gpu.temperature
            })
        
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Training metrics (if available)
        metrics = {}
        if os.path.exists('rl_training/outputs/latest_metrics.json'):
            try:
                with open('rl_training/outputs/latest_metrics.json', 'r') as f:
                    metrics = json.load(f)
            except:
                pass
        
        # Print status
        print(f"\n{'='*60}")
        print(f"Time: {timestamp}")
        print(f"CPU: {cpu_percent:.1f}%")
        print(f"RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        print(f"\nGPU Status:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']} ({gpu['name']}): {gpu['load']:.1f}% load, "
                  f"{gpu['memory_util']:.1f}% memory ({gpu['memory_used']}MB/{gpu['memory_total']}MB), "
                  f"{gpu['temperature']}°C")
        
        if metrics:
            print(f"\nTraining Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_resources()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
EOF

chmod +x rl_training/monitor_training.py

echo ""
echo "🎉 RL Training Environment Setup Complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Set your Google Cloud project:"
echo "   export GOOGLE_CLOUD_PROJECT=your-project-id"
echo ""
echo "2. Authenticate with Google Cloud:"
echo "   gcloud auth application-default login"
echo ""
echo "3. Start training:"
echo "   ./rl_training/launch_training.sh"
echo ""
echo "4. Monitor training (in another terminal):"
echo "   python rl_training/monitor_training.py"
echo ""
echo "📁 Key files created:"
echo "   - rl_training/launch_training.sh    # Main training launcher"
echo "   - rl_training/configs/ppo_config.yaml  # Training configuration"
echo "   - rl_training/monitor_training.py   # Resource monitoring"
echo ""
echo "🔗 Integration with tau-bench:"
echo "   After training, the model will be saved in rl_training/outputs/final_model"
echo "   Use deploy_model.sh to serve it with vLLM for tau-bench integration"
echo ""

# Create setup completion marker
touch rl_training/.setup_complete
echo "✅ Setup completion marker created"
