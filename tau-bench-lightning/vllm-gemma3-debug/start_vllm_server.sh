#!/bin/bash

# Start vLLM server with Gemma3:12b model
# Make sure you have vLLM installed and the model downloaded

echo "Starting vLLM server with Gemma3:12b model..."
echo "This will start the server on http://localhost:8001"
echo "Make sure you have the model available locally or specify the correct model path"
echo "Using dedicated conda environment vllm-gemma3..."

# Adjust the model name/path as needed for your setup
MODEL_NAME="google/gemma-3-12b-it"  
# Alternative: MODEL_NAME="google/gemma-2-9b-it" # Smaller and faster
PORT=8001

# Activate vllm-gemma3 environment and start vLLM server
# Use absolute interpreter to avoid base env leakage. Use GPU 1.
ENV_PY=/home/lixali_google_com/data/anaconda3/envs/vllm-gemma3/bin/python
export PYTHONNOUSERSITE=1
unset PYTHONPATH
echo "Using Python: $ENV_PY"
$ENV_PY - <<'PY'
import sys, torch
print('Server will use:', sys.executable)
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
try:
    import xformers
    print('xformers:', xformers.__version__)
except Exception as e:
    print('xformers import failed:', e)
try:
    import flash_attn
    print('flash_attn present (should be absent):', getattr(flash_attn, '__version__', 'unknown'))
except Exception as e:
    print('no flash_attn (expected):', type(e).__name__, e)
PY

CUDA_VISIBLE_DEVICES=1 $ENV_PY -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $PORT \
    --host 0.0.0.0 \
    --served-model-name gemma3:12b \
    --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --disable-custom-all-reduce

