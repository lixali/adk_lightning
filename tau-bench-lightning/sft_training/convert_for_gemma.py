#!/usr/bin/env python3
"""
Convert tau-bench data to Gemma-compatible format.

Gemma models require:
1. No system messages
2. No tool messages  
3. Strict user/assistant alternation
4. No empty content
"""

import json
import argparse

def convert_messages_for_gemma(messages):
    """Convert messages to Gemma-compatible format."""
    converted = []
    system_content = ""
    
    # Extract system message
    for msg in messages:
        if msg['role'] == 'system':
            system_content = msg['content']
            break
    
    # Process messages in order, ensuring user/assistant alternation
    last_role = None
    
    for msg in messages:
        role = msg['role']
        content = msg.get('content') or ''
        if isinstance(content, str):
            content = content.strip()
        else:
            content = str(content).strip() if content else ''
        
        if role == 'system':
            continue  # Already extracted
            
        elif role == 'user':
            # Add system content to first user message
            if system_content and not converted:
                content = f"[SYSTEM INSTRUCTIONS: {system_content[:500]}...]\n\n{content}"
                system_content = ""  # Only add once
            
            if content:  # Only add if has content
                if last_role == 'user':
                    # Two user messages in a row, combine them
                    converted[-1]['content'] += f"\n\n{content}"
                else:
                    converted.append({'role': 'user', 'content': content})
                    last_role = 'user'
                    
        elif role == 'assistant':
            assistant_content = content
            
            # Handle tool calls by converting to text
            if 'tool_calls' in msg and msg['tool_calls']:
                tool_calls_text = ""
                for tool_call in msg['tool_calls']:
                    func_name = tool_call['function']['name']
                    func_args = tool_call['function']['arguments']
                    tool_calls_text += f"\n[CALLING TOOL: {func_name} with args {func_args}]"
                
                if assistant_content:
                    assistant_content += tool_calls_text
                else:
                    assistant_content = "I'll help you with that." + tool_calls_text
            
            if assistant_content:  # Only add if has content
                if last_role == 'assistant':
                    # Two assistant messages in a row, combine them
                    converted[-1]['content'] += f"\n\n{assistant_content}"
                else:
                    converted.append({'role': 'assistant', 'content': assistant_content})
                    last_role = 'assistant'
                    
        elif role == 'tool':
            # Convert tool response to assistant message
            if content:
                tool_response = f"[TOOL RESULT: {content}]"
                if last_role == 'assistant':
                    # Add to previous assistant message
                    converted[-1]['content'] += f"\n{tool_response}"
                else:
                    # Create new assistant message
                    converted.append({'role': 'assistant', 'content': tool_response})
                    last_role = 'assistant'
    
    # Ensure alternating pattern
    filtered = []
    expected_role = 'user'
    
    for msg in converted:
        if msg['role'] == expected_role:
            filtered.append(msg)
            expected_role = 'assistant' if expected_role == 'user' else 'user'
        elif expected_role == 'user' and msg['role'] == 'assistant':
            # Missing user message, add a generic one
            filtered.append({'role': 'user', 'content': 'Please continue.'})
            filtered.append(msg)
            expected_role = 'user'
        elif expected_role == 'assistant' and msg['role'] == 'user':
            # Missing assistant message, add a generic one
            filtered.append({'role': 'assistant', 'content': 'I understand.'})
            filtered.append(msg)
            expected_role = 'assistant'
    
    return filtered

def convert_dataset(input_file, output_file):
    """Convert entire dataset."""
    converted_count = 0
    total_count = 0
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            total_count += 1
            data = json.loads(line)
            
            # Convert messages
            converted_messages = convert_messages_for_gemma(data['messages'])
            
            # Only keep if we have at least one user and one assistant message
            if len(converted_messages) >= 2:
                data['messages'] = converted_messages
                f_out.write(json.dumps(data) + '\n')
                converted_count += 1
    
    print(f"Converted {converted_count}/{total_count} conversations")
    return converted_count

def main():
    parser = argparse.ArgumentParser(description="Convert tau-bench data for Gemma")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", help="Output JSONL file")
    
    args = parser.parse_args()
    
    if not args.output:
        args.output = args.input.replace('.jsonl', '_gemma_compatible.jsonl')
    
    print(f"Converting {args.input} to {args.output}")
    count = convert_dataset(args.input, args.output)
    print(f"Done! {count} conversations saved to {args.output}")

if __name__ == "__main__":
    main()
