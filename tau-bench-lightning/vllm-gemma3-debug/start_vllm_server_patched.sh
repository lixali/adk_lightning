#!/bin/bash

# Alternative vLLM server startup with compatibility workaround
# This script patches the torch._inductor compatibility issue

echo "Starting vLLM server with compatibility workaround..."
echo "This will start the server on http://localhost:8000"

# Create a temporary Python script to patch the issue
cat > /tmp/vllm_patch.py << 'EOF'
import sys
import torch

# Patch torch._inductor.config if it doesn't exist
if not hasattr(torch._inductor, 'config'):
    print("Patching torch._inductor.config compatibility...")
    class MockConfig:
        compile_threads = 1
        cpp_wrapper = False
        triton_compile_threads = 1
    torch._inductor.config = MockConfig()

# Now import vLLM
import vllm.entrypoints.openai.api_server as api_server
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-2-12b-it")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--served-model-name", type=str, default="gemma3:12b")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=8192)
    
    args = parser.parse_args()
    
    print(f"Starting vLLM server with model: {args.model}")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    
    # Start the server
    api_server.main([
        "--model", args.model,
        "--port", str(args.port),
        "--host", args.host,
        "--served-model-name", args.served_model_name,
        "--trust-remote-code",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len)
    ])
EOF

# Run the patched script
python /tmp/vllm_patch.py \
    --model "google/gemma-3-12b-it" \
    --port 8000 \
    --host "0.0.0.0" \
    --served-model-name "gemma3:12b" \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192

echo "vLLM server stopped"

# Clean up
rm -f /tmp/vllm_patch.py
