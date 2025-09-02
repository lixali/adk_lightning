#!/usr/bin/env python3
"""
Detailed Error Analysis - Manual inspection of failed tasks
"""

import json
import sys

def inspect_failed_task(task, task_num=None):
    """Manually inspect a failed task to understand the error"""
    print(f"\n{'='*60}")
    if task_num is not None:
        print(f"TASK {task_num} (ID: {task['task_id']})")
    else:
        print(f"TASK ID: {task['task_id']}")
    print(f"{'='*60}")
    
    # Get ground truth actions
    gt_actions = task["info"]["task"]["actions"]
    print(f"📋 GROUND TRUTH ACTIONS ({len(gt_actions)} total):")
    for i, action in enumerate(gt_actions):
        print(f"  {i+1}. {action['name']}({action['kwargs']})")
    
    # Get agent's trajectory
    trajectory = task["traj"]
    print(f"\n🤖 AGENT TRAJECTORY ({len(trajectory)} messages):")
    
    agent_actions = []
    for i, msg in enumerate(trajectory):
        if msg["role"] == "user":
            print(f"  👤 User: {msg['content'][:100]}...")
        elif msg["role"] == "assistant":
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    action = {
                        "name": tool_call["function"]["name"],
                        "kwargs": json.loads(tool_call["function"]["arguments"])
                    }
                    agent_actions.append(action)
                    print(f"  🔧 Tool: {action['name']}({action['kwargs']})")
            else:
                print(f"  🤖 Assistant: {msg['content'][:100]}...")
        elif msg["role"] == "tool":
            print(f"  ⚙️  Tool result: {msg['content'][:100]}...")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Expected {len(gt_actions)} actions, agent performed {len(agent_actions)} tool calls")
    
    # Compare actions
    gt_action_names = [a["name"] for a in gt_actions]
    agent_action_names = [a["name"] for a in agent_actions]
    
    print(f"  Expected tools: {gt_action_names}")
    print(f"  Agent used tools: {agent_action_names}")
    
    return agent_actions, gt_actions

def main():
    with open('results/tool-calling-gemma3:12b-0.0_range_0-115_user-gemini-2.5-flash-llm_retail_0902061308.json', 'r') as f:
        data = json.load(f)
    
    failed_tasks = [task for task in data if task['reward'] == 0.0]
    
    print(f"Found {len(failed_tasks)} failed tasks")
    print("Inspecting first 5 failed tasks for manual analysis...")
    
    for i in range(min(5, len(failed_tasks))):
        inspect_failed_task(failed_tasks[i], i+1)
        
        if i < 4:  # Don't ask after the last one
            input("\nPress Enter to continue to next task...")

if __name__ == "__main__":
    main()
