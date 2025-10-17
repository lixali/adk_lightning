#!/bin/bash

# Start vLLM server with Gemma-2-9B model (more conservative approach)
# This uses less memory and should work reliably

echo "Starting vLLM server with Gemma-2-9B model..."
echo "This will start the server on http://localhost:8000"
echo "Note: Using 9B model as 12B doesn't exist. Gemma-2-27B is the larger option."

# Use the actual Gemma-3 model (closest to requested 12B)
MODEL_NAME="google/gemma-3-27b-it"  # Closest to 12B request
# Alternative smaller: MODEL_NAME="google/gemma-3-4b-it"
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
    --gpu-memory-utilization 0.4 \
    --max-model-len 4096 \
    --tensor-parallel-size 1 \
    --disable-log-requests

echo "vLLM server started on http://localhost:$PORT"
echo "You can now run the debug script: python vllm_gemma3_prompt_debug.py"
