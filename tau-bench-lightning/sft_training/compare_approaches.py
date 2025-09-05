#!/usr/bin/env python3
"""
Compare conversation-level vs turn-level SFT training approaches.
"""

import json

def analyze_datasets():
    print("=== SFT Training Data Analysis ===\n")
    
    # Conversation-level data
    with open('sft_training/full_sft_data_gemma_compatible.jsonl', 'r') as f:
        conv_data = [json.loads(line) for line in f]
    
    # Turn-level data  
    with open('sft_training/turn_level_sft_data_gemma.jsonl', 'r') as f:
        turn_data = [json.loads(line) for line in f]
    
    print("📊 CONVERSATION-LEVEL APPROACH:")
    print(f"  Training Examples: {len(conv_data)}")
    conv_lengths = [len(ex['messages']) for ex in conv_data]
    print(f"  Avg Messages per Example: {sum(conv_lengths)/len(conv_lengths):.1f}")
    print(f"  Total Messages: {sum(conv_lengths)}")
    
    # Count assistant messages in conversation-level
    conv_assistant_msgs = 0
    for ex in conv_data:
        conv_assistant_msgs += sum(1 for msg in ex['messages'] if msg['role'] == 'assistant')
    print(f"  Assistant Messages: {conv_assistant_msgs}")
    
    print("\n🎯 TURN-LEVEL APPROACH:")
    print(f"  Training Examples: {len(turn_data)}")
    turn_lengths = [len(ex['messages']) for ex in turn_data]
    print(f"  Avg Messages per Example: {sum(turn_lengths)/len(turn_lengths):.1f}")
    print(f"  Total Messages: {sum(turn_lengths)}")
    
    # Count assistant messages in turn-level (should be exactly len(turn_data))
    turn_assistant_msgs = 0
    for ex in turn_data:
        turn_assistant_msgs += sum(1 for msg in ex['messages'] if msg['role'] == 'assistant')
    print(f"  Assistant Messages: {turn_assistant_msgs}")
    
    print(f"\n📈 IMPROVEMENT:")
    print(f"  Data Multiplication: {len(turn_data)/len(conv_data):.1f}x more examples")
    print(f"  Training Efficiency: Each example focused on 1 assistant turn")
    print(f"  Better Gradients: More granular learning signal")
    
    print(f"\n🎯 TRAINING FOCUS:")
    print("  Conversation-level: Model learns entire conversation flow")
    print("  Turn-level: Model learns specific response patterns")
    print("  Turn-level advantage: More training signal per assistant response")
    
    # Sample comparison
    print(f"\n📝 EXAMPLE COMPARISON:")
    print("Conversation-level example:")
    conv_ex = conv_data[0]['messages']
    for i, msg in enumerate(conv_ex[:4]):
        role = msg['role']
        content = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        print(f"  {i+1}. {role}: {content}")
    print(f"  ... ({len(conv_ex)-4} more messages)")
    
    print("\nTurn-level examples (from same conversation):")
    for i, turn_ex in enumerate(turn_data[:3]):
        messages = turn_ex['messages']
        print(f"  Turn {i+1} ({len(messages)} messages):")
        last_msg = messages[-1]
        content = last_msg['content'][:60] + "..." if len(last_msg['content']) > 60 else last_msg['content']
        print(f"    Target: {content}")

if __name__ == "__main__":
    analyze_datasets()
