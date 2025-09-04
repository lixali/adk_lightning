#!/bin/bash
# Setup script to configure model cache directories in /home/lixali_google_com/data/

echo "🗂️  Setting up model cache directories in /home/lixali_google_com/data/"

# Create cache directories in the data partition
mkdir -p /home/lixali_google_com/data/model_cache/huggingface
mkdir -p /home/lixali_google_com/data/model_cache/transformers
mkdir -p /home/lixali_google_com/data/model_cache/torch
mkdir -p /home/lixali_google_com/data/model_cache/pip

# Set environment variables for this session
export HF_HOME="/home/lixali_google_com/data/model_cache/huggingface"
export TRANSFORMERS_CACHE="/home/lixali_google_com/data/model_cache/transformers"
export TORCH_HOME="/home/lixali_google_com/data/model_cache/torch"
export HF_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"
export HUGGINGFACE_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"

echo "✅ Cache directories created:"
echo "   HF_HOME: $HF_HOME"
echo "   TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE" 
echo "   TORCH_HOME: $TORCH_HOME"
echo "   HF_HUB_CACHE: $HF_HUB_CACHE"

# Clean up the existing cache to free up space
echo "🧹 Cleaning up existing cache in home directory..."
rm -rf /home/lixali_google_com/.cache/huggingface/xet/ 2>/dev/null || true
rm -rf /home/lixali_google_com/.cache/huggingface/hub/models--google--gemma* 2>/dev/null || true

echo "🔄 Available space:"
df -h /home/lixali_google_com/data/ | tail -1

echo ""
echo "✅ Cache setup complete! Environment variables are set for this session."
echo "📝 To make these permanent, add the following to your ~/.bashrc:"
echo ""
echo 'export HF_HOME="/home/lixali_google_com/data/model_cache/huggingface"'
echo 'export TRANSFORMERS_CACHE="/home/lixali_google_com/data/model_cache/transformers"'
echo 'export TORCH_HOME="/home/lixali_google_com/data/model_cache/torch"'
echo 'export HF_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"'
echo 'export HUGGINGFACE_HUB_CACHE="/home/lixali_google_com/data/model_cache/huggingface/hub"'
