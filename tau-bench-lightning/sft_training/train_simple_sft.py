#!/usr/bin/env python3
"""
Simple SFT training script for testing with smaller models.

This is a simplified version that uses google/gemma-2-2b-it and 
handles the system message issue more robustly.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer
)
from transformers.trainer_utils import set_seed
import argparse
from pathlib import Path
import logging
import random
from typing import Dict, List, Any
import wandb
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleSFTDataset(Dataset):
    """Simplified dataset that handles Gemma models properly."""
    
    def __init__(self, data_file: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.conversations = []
        with open(data_file, 'r') as f:
            for line in f:
                self.conversations.append(json.loads(line))
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self):
        return len(self.conversations)
    
    def format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation without using chat template."""
        formatted_parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg.get('content', '')
            
            if role == 'system':
                formatted_parts.append(f"System: {content}")
            elif role == 'user':
                formatted_parts.append(f"User: {content}")
            elif role == 'assistant':
                if 'tool_calls' in msg and msg['tool_calls']:
                    # Format tool calls
                    tool_calls_str = ""
                    for tool_call in msg['tool_calls']:
                        func_name = tool_call['function']['name']
                        func_args = tool_call['function']['arguments']
                        tool_calls_str += f"[TOOL_CALL: {func_name}({func_args})]"
                    
                    if content:
                        formatted_parts.append(f"Assistant: {content} {tool_calls_str}")
                    else:
                        formatted_parts.append(f"Assistant: {tool_calls_str}")
                else:
                    formatted_parts.append(f"Assistant: {content}")
            elif role == 'tool':
                formatted_parts.append(f"Tool: {content}")
        
        return "\n".join(formatted_parts)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        messages = conversation['messages']
        
        # Format conversation manually (avoid chat template issues)
        formatted_text = self.format_conversation(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels - simplified approach
        labels = input_ids.clone()
        
        # Mask tokens that aren't assistant responses
        # Simple heuristic: train on last 40% of tokens (likely assistant responses)
        mask_cutoff = int(len(input_ids) * 0.6)
        labels[:mask_cutoff] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def main():
    parser = argparse.ArgumentParser(description="Simple SFT training")
    parser.add_argument("--data-file", required=True, help="JSONL file with SFT data")
    parser.add_argument("--model-name", default="google/gemma-3-12b-it", help="Base model")
    parser.add_argument("--output-dir", default="sft_training/simple_checkpoints", help="Output directory")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--wandb-project", default="tau-bench-sft", help="Wandb project name")
    parser.add_argument("--wandb-run-name", help="Wandb run name")
    
    args = parser.parse_args()
    
    # Set custom HuggingFace cache directory with sufficient space
    cache_dir = "/home/lixali_google_com/data/huggingface_cache"
    os.environ['HF_HOME'] = cache_dir
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set random seed
    set_seed(42)
    
    # Setup wandb
    if not args.wandb_run_name:
        args.wandb_run_name = f"sft-{Path(args.model_name).name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "data_file": args.data_file,
        },
        # Don't upload models to wandb
        settings=wandb.Settings(save_code=False, code_dir=None)
    )
    
    logger.info(f"Loading model: {args.model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",  # Recommended for Gemma models
        cache_dir=cache_dir
    )
    
    # Create dataset
    full_dataset = SimpleSFTDataset(args.data_file, tokenizer, args.max_length)
    
    # Split into train and validation (90/10 split)
    total_size = len(full_dataset)
    val_size = int(total_size * 0.1)
    train_size = total_size - val_size
    
    # Shuffle indices
    indices = list(range(total_size))
    random.seed(42)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and validation datasets
    train_conversations = [full_dataset.conversations[i] for i in train_indices]
    val_conversations = [full_dataset.conversations[i] for i in val_indices]
    
    # Create separate datasets
    train_dataset = SimpleSFTDataset.__new__(SimpleSFTDataset)
    train_dataset.tokenizer = tokenizer
    train_dataset.max_length = args.max_length
    train_dataset.conversations = train_conversations
    
    val_dataset = SimpleSFTDataset.__new__(SimpleSFTDataset)
    val_dataset.tokenizer = tokenizer
    val_dataset.max_length = args.max_length
    val_dataset.conversations = val_conversations
    
    logger.info(f"Train examples: {len(train_dataset)}")
    logger.info(f"Validation examples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=50,
        learning_rate=args.learning_rate,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        # Wandb logging configuration
        report_to=["wandb"],
        run_name=args.wandb_run_name,
        logging_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Disable model uploading to wandb
        push_to_hub=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    # Log final training metrics
    wandb.log({
        "final_loss": trainer.state.log_history[-1].get("train_loss", 0),
        "total_steps": trainer.state.global_step,
        "training_completed": datetime.now().isoformat()
    })
    
    # Save final model
    final_dir = f"{args.output_dir}/final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    
    logger.info(f"Training complete! Model saved to {final_dir}")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
