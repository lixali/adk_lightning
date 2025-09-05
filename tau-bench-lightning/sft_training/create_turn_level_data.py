#!/usr/bin/env python3
"""
Create turn-level SFT training data from successful tau-bench trajectories.

Instead of using full conversations as training examples, this script creates
individual training examples for each assistant turn within a conversation.

Each training example consists of:
- Context: System message + conversation history up to current user message
- Target: Assistant response for that turn

This approach gives us more fine-grained training data and better learning.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_turn_level_examples(conversation_messages: List[Dict]) -> List[Dict]:
    """
    Extract turn-level training examples from a conversation.
    
    Args:
        conversation_messages: List of message dictionaries
        
    Returns:
        List of training examples, one per assistant turn
    """
    examples = []
    context_messages = []
    
    for i, message in enumerate(conversation_messages):
        role = message['role']
        
        if role == 'system':
            # Add system message to context
            context_messages.append(message)
            
        elif role == 'user':
            # Add user message to context
            context_messages.append(message)
            
        elif role == 'assistant':
            # Create training example: context -> assistant response
            if context_messages:  # Only if we have context
                # Create a training example
                training_example = {
                    'messages': context_messages.copy() + [message]
                }
                examples.append(training_example)
            
            # Add assistant message to context for next turn
            context_messages.append(message)
            
        elif role == 'tool':
            # Add tool response to context
            context_messages.append(message)
    
    return examples

def create_turn_level_dataset(input_file: str, output_file: str, gemma_compatible: bool = True):
    """
    Create turn-level SFT dataset from successful trajectories.
    
    Args:
        input_file: Path to successful_only.json file
        output_file: Path to output JSONL file for turn-level SFT
        gemma_compatible: Whether to make data compatible with Gemma models
    """
    logger.info(f"Loading successful trajectories from {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Processing {len(data)} successful conversations...")
    
    all_examples = []
    stats = {
        'total_conversations': len(data),
        'total_examples': 0,
        'avg_examples_per_conversation': 0,
        'assistant_turns': 0,
        'tool_calls': 0
    }
    
    for conversation_data in data:
        trajectory = conversation_data['traj']
        
        # Extract turn-level examples
        turn_examples = extract_turn_level_examples(trajectory)
        
        # Convert to Gemma-compatible format if requested
        if gemma_compatible:
            turn_examples = [convert_to_gemma_format(ex) for ex in turn_examples]
            # Filter out None examples (conversion failures)
            turn_examples = [ex for ex in turn_examples if ex is not None]
        
        all_examples.extend(turn_examples)
        
        # Update statistics
        stats['total_examples'] += len(turn_examples)
        for turn in turn_examples:
            messages = turn['messages']
            assistant_msgs = [msg for msg in messages if msg['role'] == 'assistant']
            stats['assistant_turns'] += len(assistant_msgs)
            
            for msg in assistant_msgs:
                if 'tool_calls' in msg and msg['tool_calls']:
                    stats['tool_calls'] += len(msg['tool_calls'])
    
    stats['avg_examples_per_conversation'] = stats['total_examples'] / stats['total_conversations']
    
    # Write to JSONL format
    logger.info(f"Writing {len(all_examples)} turn-level examples to {output_file}")
    with open(output_file, 'w') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    logger.info("\nTurn-Level Dataset Statistics:")
    logger.info(f"Total conversations: {stats['total_conversations']}")
    logger.info(f"Total training examples: {stats['total_examples']}")
    logger.info(f"Assistant turns: {stats['assistant_turns']}")
    logger.info(f"Tool calls: {stats['tool_calls']}")
    logger.info(f"Average examples per conversation: {stats['avg_examples_per_conversation']:.1f}")
    logger.info(f"Data multiplication factor: {stats['total_examples']/stats['total_conversations']:.1f}x")
    
    return stats

def convert_to_gemma_format(example: Dict) -> Dict:
    """Convert a turn-level example to Gemma-compatible format."""
    messages = example['messages']
    converted_messages = []
    system_content = ""
    last_role = None
    
    for msg in messages:
        role = msg['role']
        content = msg.get('content') or ''
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content).strip() if content else ''
        
        if role == 'system':
            system_content = content
            continue
            
        elif role == 'user':
            # Add system content to first user message
            if system_content and not converted_messages:
                content = f"[SYSTEM: {system_content[:300]}...]\n\n{content}"
                system_content = ""
            
            if content:
                if last_role == 'user':
                    # Combine with previous user message
                    converted_messages[-1]['content'] += f"\n\n{content}"
                else:
                    converted_messages.append({'role': 'user', 'content': content})
                    last_role = 'user'
                    
        elif role == 'assistant':
            assistant_content = content
            
            # Handle tool calls
            if 'tool_calls' in msg and msg['tool_calls']:
                tool_calls_text = ""
                for tool_call in msg['tool_calls']:
                    func_name = tool_call['function']['name']
                    func_args = tool_call['function']['arguments']
                    tool_calls_text += f"\n[TOOL: {func_name}({func_args})]"
                
                if assistant_content:
                    assistant_content += tool_calls_text
                else:
                    assistant_content = "I'll help you." + tool_calls_text
            
            if assistant_content:
                if last_role == 'assistant':
                    converted_messages[-1]['content'] += f"\n\n{assistant_content}"
                else:
                    converted_messages.append({'role': 'assistant', 'content': assistant_content})
                    last_role = 'assistant'
                    
        elif role == 'tool':
            if content:
                tool_response = f"[RESULT: {content[:200]}...]"
                if last_role == 'assistant':
                    converted_messages[-1]['content'] += f"\n{tool_response}"
                else:
                    converted_messages.append({'role': 'assistant', 'content': tool_response})
                    last_role = 'assistant'
    
    # Ensure alternating user/assistant pattern
    filtered = []
    expected_role = 'user'
    
    for msg in converted_messages:
        if msg['role'] == expected_role:
            filtered.append(msg)
            expected_role = 'assistant' if expected_role == 'user' else 'user'
        elif expected_role == 'user' and msg['role'] == 'assistant':
            # Add missing user message
            filtered.append({'role': 'user', 'content': 'Please continue.'})
            filtered.append(msg)
            expected_role = 'user'
        elif expected_role == 'assistant' and msg['role'] == 'user':
            # Add missing assistant message
            filtered.append({'role': 'assistant', 'content': 'I understand.'})
            filtered.append(msg)
            expected_role = 'assistant'
    
    # Must end with assistant message for training
    if filtered and filtered[-1]['role'] != 'assistant':
        return None
    
    # Must have at least user + assistant
    if len(filtered) < 2:
        return None
    
    return {'messages': filtered}

def main():
    parser = argparse.ArgumentParser(description="Create turn-level SFT data from tau-bench trajectories")
    parser.add_argument("--input", required=True, help="Input successful_only.json file")
    parser.add_argument("--output", help="Output JSONL file (default: turn_level_sft_data.jsonl)")
    parser.add_argument("--gemma-compatible", action="store_true", default=True, 
                       help="Make data compatible with Gemma models")
    
    args = parser.parse_args()
    
    if not args.output:
        input_path = Path(args.input)
        # Extract base name and add turn_level_sft_data suffix
        if "successful_only" in input_path.stem:
            base_name = input_path.stem.replace("_successful_only", "")
        else:
            base_name = input_path.stem
        suffix = "_gemma" if args.gemma_compatible else ""
        args.output = f"sft_training/{base_name}_turn_level_sft_data{suffix}.jsonl"
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    stats = create_turn_level_dataset(args.input, args.output, args.gemma_compatible)
    
    logger.info(f"\nTurn-level SFT dataset ready at: {args.output}")
    logger.info(f"Data increased by {stats['total_examples']/stats['total_conversations']:.1f}x compared to conversation-level!")
    
    logger.info("\nNext steps:")
    logger.info(f"python sft_training/train_simple_sft.py --data-file {args.output}")

if __name__ == "__main__":
    main()
