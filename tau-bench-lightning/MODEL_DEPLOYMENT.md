# Model Deployment Guide

Your trained model is a **LoRA adapter** (274MB) that needs to be combined with the base Gemma3-12B model for inference.

## Current Model Format

- **File**: `adapter_model.safetensors` (274MB)
- **Type**: LoRA (Low-Rank Adaptation) fine-tuned adapters
- **Format**: SafeTensors (modern, secure format - better than .bin)
- **Content**: Only the learned weight modifications, not the full model

## Deployment Options

### Option 1: Use with LiteLLM (Recommended - Keep LoRA format)

```bash
# LiteLLM can load LoRA adapters directly
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model + your adapters
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it")
model = PeftModel.from_pretrained(base_model, "./rl_training/outputs/final_model")
```

### Option 2: Merge LoRA for Ollama

```bash
# 1. Merge LoRA with base model
python rl_training/merge_lora_model.py --create_ollama_file

# 2. Create Ollama model (requires ~24GB merged model)
ollama create my-trained-gemma3 -f rl_training/outputs/merged_model/Modelfile

# 3. Use with Ollama
ollama run my-trained-gemma3
```

### Option 3: HuggingFace Transformers (Direct LoRA loading)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load your trained LoRA adapters
model = PeftModel.from_pretrained(base_model, "./rl_training/outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")

# Generate text
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Memory Requirements

| Format | Size | Memory Needed | Use Case |
|--------|------|---------------|----------|
| LoRA Adapters | 274MB | ~12GB (base model) | Development/Testing |
| Merged Model | ~24GB | ~24GB | Production/Ollama |
| Quantized (4-bit) | ~6GB | ~8GB | Resource-constrained |

## Recommendations

1. **For Development**: Use LoRA format directly with transformers/PEFT
2. **For Production with Ollama**: Merge LoRA to create full model
3. **For LiteLLM Integration**: Use LoRA format (more efficient)
4. **For API Deployment**: Consider quantization for lower memory usage

## Performance Considerations

- **LoRA Loading**: Slight overhead (~5-10ms) per inference
- **Merged Model**: No adapter overhead, but larger memory footprint
- **Quantization**: Faster inference, some quality trade-off

## Next Steps

Choose your deployment method and run the appropriate script above. The LoRA format is actually more efficient for most use cases!
