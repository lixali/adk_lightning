#!/usr/bin/env python3
"""
Prepare SFT training data from successful tau-bench trajectories.

This script converts successful tau-bench conversations into the format needed
for supervised fine-tuning (SFT) of Gemma3-12B.

The training approach:
1. Only assistant messages and tool calls are used as targets (labels)
2. System messages, user messages, and tool responses are used as context
3. Loss is only computed on assistant tokens (user/system/tool tokens are masked)
4. Tool calls are formatted in the same OpenAI function calling format used in inference
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse

def format_conversation_for_sft(traj: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a tau-bench trajectory into SFT format.
    
    Args:
        traj: List of conversation messages
        
    Returns:
        Dict with 'messages' key containing the formatted conversation
    """
    formatted_messages = []
    
    for msg in traj:
        role = msg.get('role')
        content = msg.get('content', '')
        
        if role == 'system':
            formatted_messages.append({
                'role': 'system',
                'content': content
            })
        elif role == 'user':
            formatted_messages.append({
                'role': 'user', 
                'content': content
            })
        elif role == 'assistant':
            # Assistant message - this is what we want to train on
            assistant_msg = {
                'role': 'assistant',
                'content': content
            }
            
            # Add tool calls if present
            if 'tool_calls' in msg and msg['tool_calls']:
                assistant_msg['tool_calls'] = msg['tool_calls']
                
            formatted_messages.append(assistant_msg)
            
        elif role == 'tool':
            # Tool response - used as context but not trained on
            formatted_messages.append({
                'role': 'tool',
                'content': content,
                'tool_call_id': msg.get('tool_call_id', '')
            })
    
    return {'messages': formatted_messages}

def create_sft_dataset(input_file: str, output_file: str, max_samples: int = None):
    """
    Create SFT dataset from successful trajectories.
    
    Args:
        input_file: Path to successful_only.json file
        output_file: Path to output JSONL file for SFT
        max_samples: Maximum number of samples to include (None for all)
    """
    print(f"Loading successful trajectories from {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        print(f"Using first {max_samples} samples")
    
    print(f"Processing {len(data)} successful trajectories...")
    
    sft_samples = []
    stats = {
        'total_conversations': 0,
        'total_messages': 0,
        'assistant_messages': 0,
        'tool_calls': 0,
        'avg_conversation_length': 0
    }
    
    for item in data:
        traj = item['traj']
        formatted_conv = format_conversation_for_sft(traj)
        
        # Calculate statistics
        stats['total_conversations'] += 1
        stats['total_messages'] += len(traj)
        
        assistant_count = sum(1 for msg in traj if msg.get('role') == 'assistant')
        stats['assistant_messages'] += assistant_count
        
        tool_call_count = sum(1 for msg in traj 
                             if msg.get('role') == 'assistant' and msg.get('tool_calls'))
        stats['tool_calls'] += tool_call_count
        
        sft_samples.append(formatted_conv)
    
    stats['avg_conversation_length'] = stats['total_messages'] / stats['total_conversations']
    
    # Write to JSONL format (one JSON object per line)
    print(f"Writing SFT dataset to {output_file}")
    with open(output_file, 'w') as f:
        for sample in sft_samples:
            f.write(json.dumps(sample) + '\n')
    
    print("\nDataset Statistics:")
    print(f"Total conversations: {stats['total_conversations']}")
    print(f"Total messages: {stats['total_messages']}")
    print(f"Assistant messages (training targets): {stats['assistant_messages']}")
    print(f"Tool calls: {stats['tool_calls']}")
    print(f"Average conversation length: {stats['avg_conversation_length']:.1f} messages")
    print(f"Average assistant messages per conversation: {stats['assistant_messages']/stats['total_conversations']:.1f}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT data from tau-bench trajectories")
    parser.add_argument("--input", required=True, help="Input successful_only.json file")
    parser.add_argument("--output", help="Output JSONL file (default: sft_data.jsonl)")
    parser.add_argument("--max-samples", type=int, help="Maximum number of samples to use")
    
    args = parser.parse_args()
    
    if not args.output:
        input_path = Path(args.input)
        args.output = f"sft_training/sft_data_{input_path.stem}.jsonl"
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    stats = create_sft_dataset(args.input, args.output, args.max_samples)
    
    print(f"\nSFT dataset ready at: {args.output}")
    print("\nNext steps:")
    print("1. Run the SFT training script: python sft_training/train_gemma3_sft.py")
    print("2. Use the fine-tuned model for inference with: python sft_training/sft_model_provider.py")

if __name__ == "__main__":
    main()
