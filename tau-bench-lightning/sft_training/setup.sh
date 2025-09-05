#!/bin/bash
"""
Setup script for SFT training environment.

This script sets up the environment for SFT training and integrates with tau-bench.
"""

echo "Setting up SFT training environment..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r sft_training/requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p sft_training/checkpoints
mkdir -p sft_training/data
mkdir -p sft_training/logs

echo "SFT training environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Prepare SFT data: python sft_training/prepare_sft_data.py --input results/tool-calling-gemini-2.5-pro-0.0_range_0-500_user-gemini-2.5-pro-llm_retail_train_0905052855_successful_only.json"
echo "2. Train model: python sft_training/train_gemma3_sft.py --data-file sft_training/sft_data_*.jsonl"
echo "3. Run inference: python run.py --model-provider sft --model sft-gemma3-12b ..."
