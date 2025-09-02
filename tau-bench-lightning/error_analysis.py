#!/usr/bin/env python3
"""
Error Analysis Script for Tau-Bench Results
Analyzes the four types of errors in the results file.
"""

import json
import sys
from typing import List, Dict, Any

def analyze_failed_task(task: Dict[str, Any]) -> str:
    """
    Analyze a failed task and categorize the error type.
    
    Returns one of:
    - "wrong_info": Wrong info provided to user
    - "wrong_argument": Right tool call but wrong arguments  
    - "wrong_decision": Wrong type of tool call
    - "partially_resolve": Partially resolved the request
    """
    
    task_info = task.get("info", {})
    trajectory = task.get("traj", [])
    
    # Get the actual task actions (ground truth)
    gt_actions = task_info.get("task", {}).get("actions", [])
    
    # Get agent's actions from trajectory
    agent_actions = []
    for msg in trajectory:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                agent_actions.append({
                    "name": tool_call["function"]["name"],
                    "kwargs": json.loads(tool_call["function"]["arguments"])
                })
    
    # If no actions taken, it's likely a wrong decision (didn't understand what to do)
    if not agent_actions:
        return "wrong_decision"
    
    # Get ground truth action names and agent action names
    gt_action_names = [action["name"] for action in gt_actions]
    agent_action_names = [action["name"] for action in agent_actions]
    
    # Check if agent made completely wrong tool calls
    if not any(action_name in gt_action_names for action_name in agent_action_names):
        return "wrong_decision"
    
    # Check if agent made some correct tool calls but missed some (partially resolved)
    if len(agent_actions) < len(gt_actions):
        # Check if the agent missed critical actions
        missing_actions = len(gt_actions) - len(agent_actions)
        if missing_actions > 0:
            return "partially_resolve"
    
    # Check for wrong arguments in correct tool calls
    for agent_action in agent_actions:
        for gt_action in gt_actions:
            if agent_action["name"] == gt_action["name"]:
                # Compare arguments
                if agent_action["kwargs"] != gt_action["kwargs"]:
                    return "wrong_argument"
    
    # Check trajectory for information provision issues
    for msg in trajectory:
        if msg.get("role") == "assistant" and "tool_calls" not in msg:
            # This is a response to user - check if it contains wrong info
            content = msg.get("content", "").lower()
            if any(word in content for word in ["sorry", "unable", "cannot", "don't have", "not available"]):
                return "wrong_info"
    
    # Default to partially resolved if we can't categorize clearly
    return "partially_resolve"

def main():
    if len(sys.argv) != 2:
        print("Usage: python error_analysis.py <results_file.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    total_tasks = len(data)
    failed_tasks = [task for task in data if task['reward'] == 0.0]
    successful_tasks = [task for task in data if task['reward'] == 1.0]
    
    print(f"📊 Total tasks: {total_tasks}")
    print(f"❌ Failed tasks: {len(failed_tasks)}")
    print(f"✅ Successful tasks: {len(successful_tasks)}")
    print(f"📈 Success rate: {len(successful_tasks)/total_tasks*100:.2f}%")
    print()
    
    # Analyze error types
    error_counts = {
        "wrong_info": 0,
        "wrong_argument": 0, 
        "wrong_decision": 0,
        "partially_resolve": 0
    }
    
    print("🔍 Analyzing failed tasks...")
    for i, task in enumerate(failed_tasks):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(failed_tasks)}")
        
        error_type = analyze_failed_task(task)
        error_counts[error_type] += 1
    
    print("\n📊 ERROR ANALYSIS RESULTS:")
    print("=" * 50)
    
    total_errors = len(failed_tasks)
    
    print(f"1. Wrong Info: {error_counts['wrong_info']} ({error_counts['wrong_info']/total_errors*100:.1f}%)")
    print("   - Agent omits user-required information or provides incorrect info")
    
    print(f"2. Wrong Argument: {error_counts['wrong_argument']} ({error_counts['wrong_argument']/total_errors*100:.1f}%)")  
    print("   - Right tool call but incorrect arguments")
    
    print(f"3. Wrong Decision: {error_counts['wrong_decision']} ({error_counts['wrong_decision']/total_errors*100:.1f}%)")
    print("   - Wrong type of tool call, fails to understand domain rules")
    
    print(f"4. Partially Resolve: {error_counts['partially_resolve']} ({error_counts['partially_resolve']/total_errors*100:.1f}%)")
    print("   - Omits explicit/implicit user requests")
    
    print(f"\nTotal analyzed errors: {total_errors}")

if __name__ == "__main__":
    main()
