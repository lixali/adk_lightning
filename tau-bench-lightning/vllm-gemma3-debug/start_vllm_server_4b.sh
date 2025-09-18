#!/bin/bash

# Start vLLM server with Gemma-3-4B model (smaller, more reliable)

echo "Starting vLLM server with Gemma-3-4B model..."
echo "This will start the server on http://localhost:8000"

# Use the smaller 4B model for reliable operation
MODEL_NAME="google/gemma-3-4b-it"
PORT=8000

# Clear any existing GPU processes
echo "Clearing GPU memory..."
pkill -f "vllm"
sleep 2

# Start vLLM server with conservative settings
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $PORT \
    --host 0.0.0.0 \
    --served-model-name gemma3:12b \
    --trust-remote-code \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --disable-log-requests

echo "vLLM server started on http://localhost:$PORT"
echo "You can now run the debug script: python vllm_gemma3_prompt_debug.py"
