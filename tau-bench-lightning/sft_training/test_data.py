#!/usr/bin/env python3
"""
Test script to debug data loading and chat template issues.
"""

import json
from transformers import AutoTokenizer

def test_chat_template():
    print("Testing chat template compatibility...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    
    # Load sample data
    with open("sft_training/full_sft_data.jsonl", 'r') as f:
        sample = json.loads(f.readline())
    
    messages = sample['messages']
    print(f"Sample has {len(messages)} messages")
    
    # Print message types
    role_counts = {}
    for msg in messages:
        role = msg['role']
        role_counts[role] = role_counts.get(role, 0) + 1
    
    print("Message roles:", role_counts)
    
    # Test different approaches
    print("\n1. Testing with all messages:")
    try:
        result = tokenizer.apply_chat_template(messages, tokenize=False)
        print("SUCCESS: Chat template worked")
        print("Length:", len(result))
    except Exception as e:
        print("ERROR:", str(e))
    
    print("\n2. Testing without system messages:")
    no_system = [msg for msg in messages if msg['role'] != 'system']
    try:
        result = tokenizer.apply_chat_template(no_system, tokenize=False)
        print("SUCCESS: Chat template worked without system")
        print("Length:", len(result))
    except Exception as e:
        print("ERROR:", str(e))
    
    print("\n3. Testing with converted system:")
    converted = []
    system_content = None
    
    for msg in messages:
        if msg['role'] == 'system':
            system_content = msg['content']
        elif msg['role'] == 'user':
            if system_content:
                content = f"Instructions: {system_content[:100]}...\n\nUser: {msg['content']}"
                converted.append({'role': 'user', 'content': content})
                system_content = None
            else:
                converted.append(msg)
        else:
            converted.append(msg)
    
    try:
        result = tokenizer.apply_chat_template(converted, tokenize=False)
        print("SUCCESS: Chat template worked with converted system")
        print("Length:", len(result))
        print("Sample:", result[:200] + "...")
    except Exception as e:
        print("ERROR:", str(e))

if __name__ == "__main__":
    test_chat_template()
