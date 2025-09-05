# SFT Training for Tau-Bench

This directory contains everything needed to perform Supervised Fine-Tuning (SFT) on Gemma3-12B using successful tau-bench trajectories.

## Overview

The SFT training approach:

1. **Data Preparation**: Extract successful trajectories (reward = 1.0) from tau-bench results
2. **Training**: Fine-tune Gemma3-12B on these conversations with proper masking
3. **Integration**: Use the fine-tuned model in tau-bench for evaluation

## Training Strategy

### What Gets Trained vs Masked

**TRAINED (Loss Computed):**
- ✅ Assistant responses
- ✅ Tool calls in OpenAI function calling format
- ✅ Assistant reasoning and explanations

**MASKED (Loss = 0):**
- ❌ System messages
- ❌ User messages  
- ❌ Tool responses
- ❌ Special tokens and padding

### Conversation Format

The model learns from conversations like this:

```
System: You are a helpful assistant...
User: I want to return some items
Assistant: I can help with that. [TRAINED]
Tool: search_orders result... [MASKED]
Assistant: Based on the search... [TRAINED]
```

## Quick Start

### 1. Generate Turn-Level Training Data

Create turn-level training data from successful trajectories:

```bash
# Generate turn-level data with proper naming (recommended)
python sft_training/create_turn_level_data.py \
  --input successful_only.json \
  --gemma-compatible

# This creates: sft_training/successful_only_turn_level_sft_data_gemma.jsonl
# Results in 12.2x more training examples (5,125 vs 419)
```

### 2. Run SFT Training with Validation

The training script now includes automatic validation split and wandb logging:

```bash
# Train with turn-level data (recommended for best results)
python sft_training/train_simple_sft.py \
  --data-file sft_training/successful_only_turn_level_sft_data_gemma.jsonl \
  --batch-size 4 \
  --num-epochs 3 \
  --learning-rate 5e-5

# Alternative: Train with conversation-level data
python sft_training/train_simple_sft.py \
  --data-file sft_training/successful_only_sft_data_gemma_compatible.jsonl \
  --batch-size 4 \
  --num-epochs 3 \
  --learning-rate 5e-5
```

### 3. Monitor Training Progress

- **Wandb Dashboard**: Automatic logging to `tau-bench-sft` project
- **Validation Split**: 90/10 train/validation split automatically applied
- **Validation Loss**: Logged every 50 steps for early stopping
- **Checkpoints**: Best model saved based on validation loss

```bash
# Extract successful trajectories and convert to SFT format
python sft_training/prepare_sft_data.py \
    --input results/tool-calling-gemini-2.5-pro-0.0_range_0-500_user-gemini-2.5-pro-llm_retail_train_0905052855_successful_only.json \
    --output sft_training/sft_data_gemini_pro_retail_train.jsonl
```

### 3. Train the Model

```bash
# Start SFT training
python sft_training/train_gemma3_sft.py \
    --data-file sft_training/tool-calling-gemini-2.5-pro-0.0_range_0-500_user-gemini-2.5-pro-llm_retail_train_0905052855_turn_level_sft_data_gemma.jsonl \
    --model-name google/gemma-3-12b-it \
    --output-dir sft_training/checkpoints \
    --batch-size 2 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-5 \
    --num-epochs 3
```

### 4. Use in Tau-Bench

```bash
# Run evaluation with SFT model
python run.py --agent-strategy tool-calling --env retail \
    --model sft-gemma3-12b --model-provider sft \
    --user-model gemini-2.5-pro --user-model-provider vertex_ai \
    --user-strategy llm --max-concurrency 5 \
    --task-split test --end-index 115
```

## Files Description

### Core Scripts

- **`prepare_sft_data.py`**: Converts tau-bench trajectories to SFT format
- **`train_gemma3_sft.py`**: Main SFT training script with proper masking
- **`sft_model_provider.py`**: Model provider for using SFT models in tau-bench

### Integration

- **`integrate.py`**: Integration utilities for tau-bench
- **`setup.sh`**: Environment setup script
- **`requirements.txt`**: Python dependencies

### Training Configuration

**Key Training Parameters:**
- **Base Model**: `google/gemma-3-12b-it` (or specify custom)
- **Sequence Length**: 4096 tokens
- **Batch Size**: 2 per device (with gradient accumulation)
- **Learning Rate**: 5e-5 with warmup
- **Epochs**: 3 (adjust based on data size)
- **Precision**: bfloat16 for efficiency

## Training Details

### Masking Strategy

The training uses a sophisticated masking approach:

1. **Token-level masking**: Only assistant tokens contribute to loss
2. **Message boundary detection**: Identifies assistant vs user/system content
3. **Tool call preservation**: Maintains OpenAI function calling format
4. **Special token handling**: Properly handles chat templates

### Loss Computation

```python
# Simplified loss computation
labels = input_ids.clone()
labels[~assistant_mask] = -100  # Mask non-assistant tokens
loss = CrossEntropyLoss()(logits, labels)  # Only assistant tokens contribute
```

### Model Architecture

- **Base**: Gemma2-9B-IT (instruction-tuned variant)
- **Fine-tuning**: Full parameter updates (not LoRA)
- **Memory**: Gradient checkpointing enabled
- **Optimization**: AdamW with linear warmup

## Expected Results

After SFT training, the model should:

1. **Better Tool Usage**: More accurate tool calls and parameters
2. **Improved Reasoning**: Better multi-step problem solving
3. **Domain Adaptation**: Specialized for retail/airline tasks
4. **Conversation Flow**: Natural multi-turn interactions

## Monitoring Training

Training progress is logged to:
- **Weights & Biases**: Real-time metrics and visualizations
- **Local logs**: Checkpoint saving and evaluation metrics
- **Console output**: Step-by-step progress

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or sequence length
2. **Import errors**: Ensure all dependencies installed
3. **Model not found**: Check model paths and permissions
4. **Poor performance**: Verify data quality and training hyperparameters

### Memory Requirements

- **Training**: ~40GB VRAM for Gemma2-9B with batch_size=2
- **Inference**: ~18GB VRAM for inference
- **Storage**: ~20GB for model checkpoints

## Advanced Usage

### Custom Data Sources

```bash
# Use different successful trajectory files
python sft_training/prepare_sft_data.py \
    --input your_successful_trajectories.json \
    --max-samples 1000
```

### Hyperparameter Tuning

Key parameters to adjust:
- `--learning-rate`: 1e-5 to 1e-4 range
- `--batch-size`: Based on available memory
- `--gradient-accumulation-steps`: For effective batch size
- `--num-epochs`: 2-5 epochs typically sufficient


## Integration with Tau-Bench

The SFT model integrates seamlessly with tau-bench:

1. **Provider Registration**: Added `sft` to supported providers
2. **Model Loading**: Automatic model loading and caching
3. **Response Format**: Compatible OpenAI-style responses
4. **Tool Calling**: Maintains function calling capabilities

## Troubleshooting

### Disk Space Issues

If you encounter "No space left on device" errors when downloading Gemma-3-12B (~24GB):

**Problem**: Default HuggingFace cache (`~/.cache/huggingface/`) has insufficient space.

**Solution**: The training scripts automatically use a custom cache directory:
- **Cache Location**: `/home/lixali_google_com/data/huggingface_cache/`
- **Space Available**: 4.0TB+ 
- **Automatic Setup**: Environment variables set in training scripts

**Manual Setup** (if needed):
```bash
export HF_HOME="/home/lixali_google_com/data/huggingface_cache"
export TRANSFORMERS_CACHE="/home/lixali_google_com/data/huggingface_cache"
```

### Common Issues
1. **CUDA OOM**: Reduce batch size or enable gradient accumulation
2. **Chat template errors**: Use `--gemma-compatible` flag for data preparation
3. **Model download timeout**: Check internet connection, downloads resume automatically

### Memory Requirements
- **Gemma-3-12B**: ~40GB VRAM (batch size 1-2)
- **Gradient accumulation**: Maintains effective batch size with less memory
- **Model caching**: ~24GB disk space for model weights

## Performance Expectations

Expected improvements over base model:
- **Success Rate**: 10-20% improvement on test tasks
- **Tool Accuracy**: Better parameter selection and tool usage
- **Conversation Quality**: More natural and effective interactions
- **Domain Knowledge**: Better understanding of retail/airline domains

## Next Steps

After successful training:

1. **Evaluate on test set**: Compare against baseline models
2. **Error analysis**: Identify remaining failure modes  
3. **Iterative improvement**: Collect more data and retrain
4. **Deployment**: Use in production tau-bench evaluations
