# File: /vllm-gemma3-debug/vllm-gemma3-debug/src/vllm_gemma3_prompt_debug.py

import argparse
import json
import os
import random
import time
import multiprocessing
from datetime import datetime
from typing import List
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass



# Add the tau-bench-lightning directory to path to find tau_bench module
tau_bench_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if os.path.exists(tau_bench_path) and tau_bench_path not in sys.path:
    sys.path.insert(0, tau_bench_path)

# Add sft_training to path for SFT model integration (if needed)
sft_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sft_training")
if os.path.exists(sft_path) and sft_path not in sys.path:
    sys.path.insert(0, sft_path)


from tau_bench.agents.base import Agent
from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult
from tau_bench.run import display_metrics
from litellm import completion

from agents.vllm_tool_calling_agent import VLLMToolCallingAgent

# Configuration setup based on the specified parameters
@dataclass
class Config:
    agent_strategy: str = "tool-calling"
    env: str = "retail"
    model: str = "gemma3:12b"
    model_provider: str = "vllm"
    user_model: str = "gemini-2.5-pro"
    user_model_provider: str = "vertex_ai"
    user_strategy: str = "llm"
    max_concurrency: int = 1
    task_id: int = 0  # Set to 0 to run task 0 by default
    user_temperature: float = 0.0
    task_split: str = "test"
    end_index: int = -1
    start_index: int = 0
    task_ids: List[int] = None
    num_trials: int = 1
    shuffle: bool = False

def parse_args():
    parser = argparse.ArgumentParser(description="Run vllm gemma3 debug with tau-bench")
    parser.add_argument("--max-concurrency", type=int, default=1,
                        help="Maximum number of concurrent tasks (default: 1)")
    parser.add_argument("--end-index", type=int, default=-1,
                        help="End index for task range, -1 for all tasks (default: -1)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index for task range (default: 0)")
    parser.add_argument("--task-id", type=int, default=0,
                        help="Single task ID to run (overrides start/end index)")
    parser.add_argument("--model", type=str, default="gemma3:12b",
                        help="Model to use (default: gemma3:12b)")
    parser.add_argument("--model-provider", type=str, default="vllm",
                        help="Model provider (default: vllm)")
    parser.add_argument("--user-model", type=str, default="gemini-2.5-pro",
                        help="User model (default: gemini-2.5-pro)")
    parser.add_argument("--user-model-provider", type=str, default="vertex_ai",
                        help="User model provider (default: vertex_ai)")
    parser.add_argument("--num-trials", type=int, default=1,
                        help="Number of trials per task (default: 1)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle task order")
    return parser.parse_args()

# Parse command line arguments and update config
args = parse_args()
config = Config()
config.max_concurrency = args.max_concurrency
config.end_index = args.end_index
config.start_index = args.start_index
config.task_id = 0  # Force task_id to 0 for debugging
config.model = args.model
config.model_provider = args.model_provider
config.user_model = args.user_model
config.user_model_provider = args.user_model_provider
config.num_trials = args.num_trials
config.shuffle = args.shuffle

# Set up task_ids to specifically run task 0
config.task_ids = [0]  # Force to run only task 0

print(f"🔧 DEBUG: Configuration setup (FORCED TO TASK 0):")
print(f"   task_id: {config.task_id}")
print(f"   task_ids: {config.task_ids}")
print(f"   start_index: {config.start_index}")
print(f"   end_index: {config.end_index}")
print(f"   max_concurrency: {config.max_concurrency}")
print(f"   env: {config.env}")
print(f"   task_split: {config.task_split}")
print(f"   model: {config.model}")
print(f"   model_provider: {config.model_provider}")
print(f"   num_trials: {config.num_trials}")
print(f"   shuffle: {config.shuffle}")

# Initialize environment
env = get_env(
    config.env,
    user_strategy=config.user_strategy,
    user_model=config.user_model,
    user_provider=config.user_model_provider,
    user_temperature=config.user_temperature,
    task_split=config.task_split,
)

# Agent factory function
def agent_factory(tools_info, wiki, config):
    return VLLMToolCallingAgent(
        tools_info=tools_info,
        wiki=wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=0.0
    )

# Checkpoint path setup - save to log/ directory
import os
log_dir = os.path.join(os.path.dirname(__file__), "log")
os.makedirs(log_dir, exist_ok=True)
ckpt_path = os.path.join(log_dir, f"vllm_gemma3_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

# Start timing
start_time = time.time()

# Read retail_tools_info.md and append to env.wiki
retail_tools_path = "/home/lixali_google_com/data/adk-lightning/tau-bench-lightning/retail_tools_info.md"
try:
    with open(retail_tools_path, "r") as f:
        retail_tools_content = f.read()
    enhanced_wiki = env.wiki + "\n\n" + "="*80 + "\n" + "AVAILABLE TOOLS DOCUMENTATION:\n" + "="*80 + "\n\n" + retail_tools_content
    print(f"🔧 DEBUG: Enhanced wiki with retail tools info ({len(retail_tools_content)} chars added)")
except FileNotFoundError:
    print(f"⚠️  WARNING: retail_tools_info.md not found at {retail_tools_path}, using original wiki")
    enhanced_wiki = env.wiki

agent = agent_factory(
    wiki=enhanced_wiki,
    config=config,
    tools_info=None,  # Set to None to disable tools
)

end_index = (
    len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
)
results: List[EnvRunResult] = []

if config.task_ids and len(config.task_ids) > 0:
    print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
else:
    print(f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})")

for i in range(config.num_trials):
    if config.task_ids and len(config.task_ids) > 0:
        idxs = config.task_ids
    else:
        idxs = list(range(config.start_index, end_index))
    if config.shuffle:
        random.shuffle(idxs)

    for idx in idxs:
        isolated_env = get_env(
            config.env,
            user_strategy=config.user_strategy,
            user_model=config.user_model,
            task_split=config.task_split,
            user_provider=config.user_model_provider,
            user_temperature=config.user_temperature,
            task_index=idx,
        )

        print(f"Running task {idx}")
        try:
            res = agent.solve(
                env=isolated_env,
                task_index=idx,
            )
            result = EnvRunResult(
                task_id=idx,
                reward=res.reward,
                info=res.info,
                traj=res.messages,
                trial=i,
            )
        except Exception as e:
            result = EnvRunResult(
                task_id=idx,
                reward=0.0,
                info={"error": str(e)},
                traj=[],
                trial=i,
            )
        print("✅" if result.reward == 1 else "❌", f"task_id={idx}", result.info)
        print("-----")
        results.append(result)
        with open(ckpt_path, "a") as f:
            json.dump(result.model_dump(), f, indent=2)
            f.write("\n")

metrics = display_metrics(results)

# Calculate total runtime
end_time = time.time()
total_runtime = end_time - start_time
runtime_minutes = total_runtime / 60
runtime_hours = total_runtime / 3600

# Print runtime to terminal
print(f"\n⏱️  Total Runtime: {total_runtime:.2f} seconds ({runtime_minutes:.2f} minutes, {runtime_hours:.2f} hours)")