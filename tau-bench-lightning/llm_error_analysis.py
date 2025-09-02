#!/usr/bin/env python3
"""
LLM-based Error Analysis Script for Tau-Bench Results
Uses Gemini-2.5-pro to analyze the four types of errors in failed tasks.
"""

import json
import sys
import os
from typing import List, Dict, Any
import vertexai
from vertexai.preview.generative_models import GenerativeModel


def setup_vertex_ai():
    """Initialize Vertex AI"""
    try:
        # Initialize Vertex AI
        vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location="us-central1")
        model = GenerativeModel("gemini-2.5-pro")
        return model
    except Exception as e:
        print(f"Error setting up Vertex AI: {e}")
        print("Make sure you have set up Google Cloud credentials and GOOGLE_CLOUD_PROJECT environment variable")
        return None

def analyze_failed_task_with_llm(task: Dict[str, Any], model) -> str:
    """
    Use Gemini-2.5-pro to analyze a failed task and categorize the error type.
    """
    
    task_info = task.get("info", {})
    trajectory = task.get("traj", [])
    reward = task.get("reward", 0)
    
    # Get the ground truth task
    gt_task = task_info.get("task", {})
    gt_actions = gt_task.get("actions", [])
    gt_instruction = gt_task.get("instruction", "")
    gt_outputs = gt_task.get("outputs", [])
    
    # Format the conversation trajectory
    conversation = []
    for msg in trajectory:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "assistant" and msg.get("tool_calls"):
            # Format tool calls
            tool_calls = []
            for tool_call in msg["tool_calls"]:
                tool_calls.append({
                    "name": tool_call["function"]["name"],
                    "arguments": tool_call["function"]["arguments"]
                })
            conversation.append(f"ASSISTANT (tool calls): {json.dumps(tool_calls, indent=2)}")
        else:
            conversation.append(f"{role.upper()}: {content}")
    
    conversation_text = "\n".join(conversation)
    
    # Create the prompt for LLM analysis
    prompt = f"""
You are an expert at analyzing AI assistant failures in customer service scenarios. Please analyze the following failed task and categorize the error into exactly ONE of these four types:

**ERROR TYPES:**
1. **Wrong info** - The agent omits user-required information (e.g., user asks for tracking ID but agent doesn't provide it), calculates wrong information (e.g., wrong total price), or provides incorrect information that causes the user request to diverge.

2. **Wrong argument** - The agent makes the right type of tool call(s) but fills in one or more arguments incorrectly.

3. **Wrong decision** - The agent fails to understand the domain-specific knowledge or rules and makes the wrong type of tool call.

4. **Partially resolve** - The agent omits explicit user requests at the beginning of the conversation, or omits implicit actions. Sometimes this hints at the need for better long-context and memory capabilities.

**TASK DETAILS:**
- **Task ID:** {task.get('task_id', 'unknown')}
- **Reward:** {reward} (0.0 means failure)
- **User Instruction:** {gt_instruction}
- **Expected Actions:** {json.dumps(gt_actions, indent=2)}
- **Expected Outputs:** {gt_outputs}

**CONVERSATION TRAJECTORY:**
{conversation_text}

**ANALYSIS INSTRUCTIONS:**
1. Carefully read the user instruction and expected actions/outputs
2. Follow the conversation trajectory to understand what the agent did
3. Identify where and why the agent failed
4. Categorize the error into exactly ONE of the four types above

Please respond with only the error type name (one of: "wrong_info", "wrong_argument", "wrong_decision", "partially_resolve") followed by a brief explanation of why you chose this category.

Format your response as:
ERROR_TYPE: [one of the four types]
EXPLANATION: [brief explanation of why this error type was chosen]
"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract the error type from the response
        lines = response_text.split('\n')
        error_type = None
        explanation = ""
        
        for line in lines:
            if line.startswith("ERROR_TYPE:"):
                error_type = line.split(":", 1)[1].strip().lower()
            elif line.startswith("EXPLANATION:"):
                explanation = line.split(":", 1)[1].strip()
        
        # Validate error type
        valid_types = ["wrong_info", "wrong_argument", "wrong_decision", "partially_resolve"]
        if error_type not in valid_types:
            print(f"Invalid error type '{error_type}' for task {task.get('task_id')}. Using 'wrong_decision' as fallback.")
            error_type = "wrong_decision"
        
        return error_type, explanation
        
    except Exception as e:
        print(f"Error analyzing task {task.get('task_id')} with LLM: {e}")
        return "wrong_decision", f"LLM analysis failed: {e}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python llm_error_analysis.py <results_file.json>")
        print("Example: python llm_error_analysis.py results/tool-calling-gemma3:12b-0.0_range_0-115_user-gemini-2.5-flash-llm_retail_0902061308.json")
        print("Note: The script will automatically analyze ALL failed tasks in the file")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Setup Vertex AI
    print("🔧 Setting up Vertex AI...")
    model = setup_vertex_ai()
    if model is None:
        sys.exit(1)
    
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
    
    # Analyze error types using LLM
    error_counts = {
        "wrong_info": 0,
        "wrong_argument": 0, 
        "wrong_decision": 0,
        "partially_resolve": 0
    }
    
    error_examples = {
        "wrong_info": [],
        "wrong_argument": [], 
        "wrong_decision": [],
        "partially_resolve": []
    }
    
    print("🤖 Analyzing failed tasks with Gemini-2.5-pro...")
    
    # Analyze all failed tasks
    tasks_to_analyze = len(failed_tasks)
    print(f"Analyzing ALL {tasks_to_analyze} failed tasks...")
    
    for i, task in enumerate(failed_tasks):
        print(f"Progress: {i+1}/{tasks_to_analyze} - Task ID: {task.get('task_id')}")
        
        error_type, explanation = analyze_failed_task_with_llm(task, model)
        error_counts[error_type] += 1
        
        # Store examples (first 3 of each type)
        if len(error_examples[error_type]) < 3:
            error_examples[error_type].append({
                "task_id": task.get('task_id'),
                "explanation": explanation,
                "instruction": task.get("info", {}).get("task", {}).get("instruction", "")[:200] + "..."
            })
    
    print("\n🤖 LLM-BASED ERROR ANALYSIS RESULTS:")
    print("=" * 60)
    
    total_analyzed = sum(error_counts.values())
    
    print(f"1. Wrong Info: {error_counts['wrong_info']} ({error_counts['wrong_info']/total_analyzed*100:.1f}%)")
    print("   - Agent omits user-required information or provides incorrect info")
    for example in error_examples['wrong_info']:
        print(f"     • Task {example['task_id']}: {example['explanation']}")
    
    print(f"\n2. Wrong Argument: {error_counts['wrong_argument']} ({error_counts['wrong_argument']/total_analyzed*100:.1f}%)")  
    print("   - Right tool call but incorrect arguments")
    for example in error_examples['wrong_argument']:
        print(f"     • Task {example['task_id']}: {example['explanation']}")
    
    print(f"\n3. Wrong Decision: {error_counts['wrong_decision']} ({error_counts['wrong_decision']/total_analyzed*100:.1f}%)")
    print("   - Wrong type of tool call, fails to understand domain rules")
    for example in error_examples['wrong_decision']:
        print(f"     • Task {example['task_id']}: {example['explanation']}")
    
    print(f"\n4. Partially Resolve: {error_counts['partially_resolve']} ({error_counts['partially_resolve']/total_analyzed*100:.1f}%)")
    print("   - Omits explicit/implicit user requests")
    for example in error_examples['partially_resolve']:
        print(f"     • Task {example['task_id']}: {example['explanation']}")
    
    print(f"\nTotal analyzed errors: {total_analyzed}")
    print(f"Analyzed ALL {len(failed_tasks)} failed tasks")
    
    # Save detailed results
    output_file = results_file.replace('.json', '_llm_error_analysis.json')
    detailed_results = {
        "summary": {
            "total_tasks": total_tasks,
            "failed_tasks": len(failed_tasks),
            "successful_tasks": len(successful_tasks),
            "success_rate": len(successful_tasks)/total_tasks*100,
            "analyzed_tasks": total_analyzed
        },
        "error_counts": error_counts,
        "error_examples": error_examples
    }
    
    with open(output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
