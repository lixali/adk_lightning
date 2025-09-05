#!/usr/bin/env python3
"""
Script to filter JSON results and extract only tasks with reward = 1.0
"""
import json
import sys
import os
from pathlib import Path

def filter_successful_tasks(input_file, output_file=None):
    """
    Parse JSON file and extract only dictionaries with "reward": 1.0
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file (optional)
    """
    # Read the input JSON file
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{input_file}': {e}")
        return
    
    # Ensure data is a list
    if not isinstance(data, list):
        print(f"Error: Expected a list of dictionaries, got {type(data)}")
        return
    
    # Filter for tasks with reward = 1.0
    successful_tasks = []
    for item in data:
        if isinstance(item, dict) and item.get("reward") == 1.0:
            successful_tasks.append(item)
    
    print(f"Found {len(successful_tasks)} successful tasks out of {len(data)} total tasks")
    print(f"Success rate: {len(successful_tasks)/len(data)*100:.1f}%")
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_successful_only.json"
    
    # Write the filtered results to output file
    try:
        with open(output_file, 'w') as f:
            json.dump(successful_tasks, f, indent=2)
        print(f"Successfully wrote {len(successful_tasks)} successful tasks to '{output_file}'")
    except Exception as e:
        print(f"Error writing to file '{output_file}': {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_successful_tasks.py <input_json_file> [output_json_file]")
        print("Example: python filter_successful_tasks.py results/my_results.json")
        print("Example: python filter_successful_tasks.py results/my_results.json filtered_results.json")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    filter_successful_tasks(input_file, output_file)

if __name__ == "__main__":
    main()
