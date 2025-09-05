#!/usr/bin/env python3
"""
SFT Training Summary and Quick Start Guide

This script provides a comprehensive overview of the SFT training setup
and shows how to run the complete pipeline.
"""

import os
import json
from pathlib import Path

def print_banner(text):
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def main():
    print_banner("TAU-BENCH SFT TRAINING SETUP")
    
    print("\n📁 Directory Structure:")
    sft_dir = Path("sft_training")
    if sft_dir.exists():
        for file in sorted(sft_dir.iterdir()):
            if file.is_file():
                size = file.stat().st_size
                if size > 1024*1024:
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"  ✓ {file.name} ({size_str})")
    
    print_banner("TRAINING DATA STATISTICS")
    
    # Check if full dataset exists
    full_data_path = "sft_training/full_sft_data.jsonl"
    if os.path.exists(full_data_path):
        with open(full_data_path, 'r') as f:
            sample_count = sum(1 for _ in f)
        
        # Get statistics from a sample
        with open(full_data_path, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
            msg_count = len(sample['messages'])
            assistant_msgs = sum(1 for msg in sample['messages'] if msg['role'] == 'assistant')
        
        print(f"  📊 Training Conversations: {sample_count}")
        print(f"  📝 Avg Messages per Conversation: ~{msg_count}")
        print(f"  🤖 Avg Assistant Messages: ~{assistant_msgs}")
        print(f"  📋 File Size: {os.path.getsize(full_data_path)/(1024*1024):.1f}MB")
        
        # Sample conversation structure
        print("\n📋 Sample Conversation Structure:")
        for i, msg in enumerate(sample['messages'][:5]):
            role = msg['role']
            content = msg.get('content', '')
            has_tools = 'tool_calls' in msg
            
            content_preview = content[:50] + "..." if len(content) > 50 else content
            tool_info = " [WITH_TOOLS]" if has_tools else ""
            print(f"  {i+1}. {role.upper()}: {content_preview}{tool_info}")
        
        if len(sample['messages']) > 5:
            print(f"  ... ({len(sample['messages'])-5} more messages)")
    
    print_banner("TRAINING MASKING STRATEGY")
    print("""
  🎯 TRAINED (Loss Computed):
     ✅ Assistant responses
     ✅ Tool calls in OpenAI format
     ✅ Assistant reasoning
  
  🚫 MASKED (Loss = 0):
     ❌ System messages
     ❌ User messages  
     ❌ Tool responses
     ❌ Special tokens
    """)
    
    print_banner("QUICK START COMMANDS")
    
    print("\n1️⃣ Setup Environment:")
    print("   bash sft_training/setup.sh")
    
    print("\n2️⃣ Train Model (Basic):")
    print("   python sft_training/train_gemma3_sft.py \\")
    print("     --data-file sft_training/full_sft_data.jsonl \\")
    print("     --model-name google/gemma-2-9b-it \\")
    print("     --batch-size 2 --gradient-accumulation-steps 8 \\")
    print("     --num-epochs 3 --learning-rate 5e-5")
    
    print("\n3️⃣ Train Model (Memory Optimized):")
    print("   python sft_training/train_gemma3_sft.py \\")
    print("     --data-file sft_training/full_sft_data.jsonl \\")
    print("     --model-name google/gemma-2-2b-it \\")
    print("     --batch-size 1 --gradient-accumulation-steps 16 \\")
    print("     --num-epochs 3 --max-length 2048")
    
    print("\n4️⃣ Run Evaluation:")
    print("   python run.py --agent-strategy tool-calling --env retail \\")
    print("     --model sft-gemma3-12b --model-provider sft \\")
    print("     --user-model gemini-2.5-pro --user-model-provider vertex_ai \\")
    print("     --user-strategy llm --task-split test --end-index 115")
    
    print_banner("MEMORY REQUIREMENTS")
    print("""
  🖥️  Training Gemma2-9B:
      • GPU Memory: ~40GB VRAM (batch_size=2)
      • System RAM: ~32GB recommended
      • Storage: ~20GB for checkpoints
  
  ⚡ Memory Optimizations:
      • Use smaller model: google/gemma-2-2b-it
      • Reduce batch_size to 1
      • Reduce max_length to 2048
      • Enable gradient_checkpointing (default)
    """)
    
    print_banner("EXPECTED IMPROVEMENTS")
    print("""
  📈 Performance Gains:
      • Success Rate: +10-20% on test tasks
      • Tool Usage: Better parameter accuracy
      • Conversation: More natural interactions
      • Domain Knowledge: Retail/airline expertise
  
  🎯 Training Targets:
      • Tool calling accuracy
      • Multi-turn conversation flow
      • Domain-specific reasoning
      • Instruction following
    """)
    
    print_banner("NEXT STEPS")
    print("""
  1. Install dependencies: pip install -r sft_training/requirements.txt
  2. Start training with your preferred configuration
  3. Monitor training via Weights & Biases
  4. Evaluate on tau-bench test set
  5. Compare against baseline models
  6. Iterate and improve based on results
    """)
    
    print("\n" + "="*60)
    print(" Ready to start SFT training! 🚀")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
