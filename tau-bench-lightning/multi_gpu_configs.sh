#!/bin/bash
# multi_gpu_configs.sh - Different configurations for various models

echo "🎯 Multi-GPU Configurations for tau-bench"
echo

# Configuration 1: Large Model (70B) - Use all 8 GPUs
start_large_model() {
    echo "🔥 Starting Large Model (70B parameters) on 8 H100s"
    python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 8 \
        --gpu-memory-utilization 0.9 \
        --served-model-name llama-3.1-70b \
        --max-model-len 4096 &
}

# Configuration 2: Medium Model (7B) - Use 2 GPUs, leave 6 for other tasks
start_medium_model() {
    echo "⚡ Starting Medium Model (7B parameters) on 2 H100s"
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --host 0.0.0.0 \
        --port 8001 \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.8 \
        --served-model-name llama-3.1-8b \
        --max-model-len 8192 &
}

# Configuration 3: Multiple Small Models - Distribute across GPUs
start_distributed_models() {
    echo "🌐 Starting Multiple Models Distributed Across GPUs"
    
    # Model 1 on GPU 0-1
    CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
        --model microsoft/DialoGPT-medium \
        --host 0.0.0.0 \
        --port 8002 \
        --tensor-parallel-size 2 \
        --served-model-name dialogpt-medium &
    
    # Model 2 on GPU 2-3  
    CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
        --model microsoft/DialoGPT-large \
        --host 0.0.0.0 \
        --port 8003 \
        --tensor-parallel-size 2 \
        --served-model-name dialogpt-large &
        
    # Model 3 on GPU 4-5
    CUDA_VISIBLE_DEVICES=4,5 python -m vllm.entrypoints.openai.api_server \
        --model codellama/CodeLlama-7b-Instruct-hf \
        --host 0.0.0.0 \
        --port 8004 \
        --tensor-parallel-size 2 \
        --served-model-name codellama-7b &
        
    # Reserve GPU 6-7 for other tasks
    echo "📋 Reserved GPUs 6-7 for additional workloads"
}

# Configuration 4: Optimal for tau-bench customer service
start_customer_service_model() {
    echo "🎧 Starting Customer Service Optimized Model on 4 H100s"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --host 0.0.0.0 \
        --port 8005 \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.85 \
        --served-model-name customer-service-llama \
        --max-model-len 6144 \
        --enable-chunked-prefill &
}

# Menu
echo "Choose configuration:"
echo "1) Large Model (70B) - 8 GPUs"
echo "2) Medium Model (8B) - 2 GPUs" 
echo "3) Distributed Models - Multiple models across GPUs"
echo "4) Customer Service Optimized - 4 GPUs"
echo

read -p "Enter choice (1-4): " choice

case $choice in
    1) start_large_model ;;
    2) start_medium_model ;;
    3) start_distributed_models ;;
    4) start_customer_service_model ;;
    *) echo "Invalid choice" ;;
esac

echo "🎉 Server(s) starting... Check with 'ps aux | grep vllm'"
