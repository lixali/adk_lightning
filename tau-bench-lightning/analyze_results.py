#!/usr/bin/env python3
"""
Analysis script to calculate Pass@k metrics from tau-bench results files.
Usage: python analyze_results.py <results_file.json>
"""

import json
import sys
from typing import List, Dict
from math import comb

def is_successful(reward: float) -> bool:
    """Check if a task was successful (reward ≈ 1.0)"""
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

def calculate_metrics(results_file: str):
    """Calculate and display Pass@k metrics from results file"""
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print(f"📊 Analyzing results from: {results_file}")
    print(f"📝 Total entries: {len(results)}")
    
    # Extract basic metrics
    rewards = [r['reward'] for r in results]
    avg_reward = sum(rewards) / len(rewards)
    
    # Count trials and successful tasks
    num_trials = len(set([r.get('trial', 0) for r in results]))
    if num_trials == 0:
        num_trials = 1  # Default to 1 trial if not specified
    
    print(f"🔄 Number of trials: {num_trials}")
    print(f"🏆 Average reward: {avg_reward:.4f} ({avg_reward*100:.2f}% pass rate)")
    
    # Calculate success count per task_id (for Pass@k)
    c_per_task_id: Dict[int, int] = {}
    for result in results:
        task_id = result['task_id']
        if task_id not in c_per_task_id:
            c_per_task_id[task_id] = 1 if is_successful(result['reward']) else 0
        else:
            c_per_task_id[task_id] += 1 if is_successful(result['reward']) else 0
    
    print(f"📋 Unique tasks: {len(c_per_task_id)}")
    
    # Calculate Pass@k metrics
    pass_hat_ks: Dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            if c >= k:
                sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
            # If c < k, the probability is 0, so we don't add anything
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    
    print("\n📈 Pass@k Metrics:")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  Pass@{k}: {pass_hat_k:.4f} ({pass_hat_k*100:.2f}%)")
    
    # Additional statistics
    successful_tasks = sum(1 for c in c_per_task_id.values() if c > 0)
    print(f"\n📊 Additional Stats:")
    print(f"  Tasks with at least 1 success: {successful_tasks}/{len(c_per_task_id)} ({successful_tasks/len(c_per_task_id)*100:.2f}%)")
    
    if num_trials > 1:
        perfectly_solved = sum(1 for c in c_per_task_id.values() if c == num_trials)
        print(f"  Tasks solved in all trials: {perfectly_solved}/{len(c_per_task_id)} ({perfectly_solved/len(c_per_task_id)*100:.2f}%)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_results.py <results_file.json>")
        print("Example: python analyze_results.py results/tool-calling-gemini-2.5-flash-0.0_range_0--1_user-gemini-2.5-flash-llm_0826161338.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    try:
        calculate_metrics(results_file)
    except FileNotFoundError:
        print(f"❌ Error: File '{results_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{results_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
