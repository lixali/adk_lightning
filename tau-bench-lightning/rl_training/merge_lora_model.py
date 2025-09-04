#!/usr/bin/env python3
"""
Merge LoRA adapters with base model for deployment
This creates a full model that can be used with Ollama or other inference engines
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def merge_lora_model(
    base_model_path: str = "google/gemma-3-12b-it",
    adapter_path: str = "./rl_training/outputs/final_model",
    output_path: str = "./rl_training/outputs/merged_model",
    device_map: str = "auto"
):
    """
    Merge LoRA adapters with base model
    """
    print(f"🔄 Loading base model: {base_model_path}")
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    
    print(f"🔄 Loading LoRA adapters: {adapter_path}")
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("🔄 Merging LoRA adapters with base model...")
    
    # Merge adapters into base model
    merged_model = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    merged_model.save_pretrained(
        output_path,
        save_method="safetensors",
        max_shard_size="5GB"
    )
    
    # Load and save tokenizer
    print("💾 Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Model merging completed!")
    print(f"📁 Merged model saved to: {output_path}")
    print("\n🚀 Usage options:")
    print("1. For Ollama: Convert to GGUF format using ollama create")
    print("2. For LiteLLM: Use the merged model path directly")
    print("3. For Hugging Face: Load with AutoModelForCausalLM.from_pretrained()")

def create_ollama_modelfile(output_path: str):
    """Create Ollama Modelfile for the merged model"""
    modelfile_content = f"""FROM {output_path}

# Set custom parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# Set custom template (adjust based on your training format)
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}{{{{ .Response }}}}<|im_end|>\"\"\"

# Set stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
    
    modelfile_path = os.path.join(output_path, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"📝 Ollama Modelfile created: {modelfile_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters with base model")
    parser.add_argument("--base_model", default="google/gemma-3-12b-it", help="Base model path")
    parser.add_argument("--adapter_path", default="./rl_training/outputs/final_model", help="LoRA adapter path")
    parser.add_argument("--output_path", default="./rl_training/outputs/merged_model", help="Output path for merged model")
    parser.add_argument("--create_ollama_file", action="store_true", help="Create Ollama Modelfile")
    
    args = parser.parse_args()
    
    # Merge model
    merge_lora_model(
        base_model_path=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path
    )
    
    # Create Ollama Modelfile if requested
    if args.create_ollama_file:
        create_ollama_modelfile(args.output_path)
