#!/bin/bash
# start_vllm_server.sh - Script to start vLLM server using all 8 H100 GPUs

# Configuration
MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"  # Example model
PORT=8000
HOST="0.0.0.0"
TENSOR_PARALLEL_SIZE=8  # Use all 8 H100 GPUs

echo "🚀 Starting vLLM server with 8 H100 GPUs..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"

# Start vLLM server with multi-GPU support
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization 0.9 \
    --enable-chunked-prefill \
    --max-model-len 4096 \
    --served-model-name llama-3.1-70b \
    --chat-template template_chatml.jinja

echo "✅ vLLM server started on http://localhost:$PORT"
echo "📊 Using $TENSOR_PARALLEL_SIZE H100 GPUs"
echo "🔍 Check status: curl http://localhost:$PORT/v1/models"
