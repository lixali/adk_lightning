####### cell #######

import json
from litellm import completion
import sys
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import argparse
from dataclasses import dataclass

# Add sft_training to path for SFT model integration
sft_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sft_training")
if os.path.exists(sft_path) and sft_path not in sys.path:
    sys.path.insert(0, sft_path)

from tau_bench.agents.base import Agent
from tau_bench.agents import debug_message_to_action
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME

####### cell #######

class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: Optional[List[Dict[str, Any]]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
    ):
        # DEBUGGING: Select specific tools for exchange scenario
        relevant_tool_names = [
            'find_user_id_by_name_zip', 
            'get_order_details', 
            'get_product_details',
            'exchange_delivered_order_items'
        ]
        
        # Handle None tools_info
        if tools_info is None:
            self.tools_info = []
        else:
            # self.tools_info = [tool for tool in tools_info if tool['function']['name'] in relevant_tool_names]
            self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        
        print(f"🔧 DEBUG: Agent initialized with:")
        print(f"   Model: {model}")
        print(f"   Provider: {provider}")
        print(f"   Temperature: {temperature}")
        print(f"   Wiki length: {len(wiki)}")
        print(f"   Tools count: {len(tools_info) if tools_info else 0} -> LIMITED TO: {len(self.tools_info)}")
        if self.tools_info:
            print(f"   Available tools: {[tool['function']['name'] for tool in self.tools_info]}")
        else:
            print(f"   Available tools: None (no tools provided)")
        
        # Handle custom providers by mapping them to litellm providers
        if provider == "gemma":
            # Auto-detect the appropriate litellm provider based on model name
            if model.startswith("gemma:") or ":" in model:
                self.litellm_provider = "ollama"
                self.litellm_model = model  # Keep the model as is for Ollama
                print(f"🔧 DEBUG: Mapped to Ollama - {model}")
            elif model.startswith("vllm/") or model.startswith("/"):
                # vLLM deployment (local path or vllm/ prefix)
                self.litellm_provider = "vllm"  
                self.litellm_model = model.replace("vllm/", "") if model.startswith("vllm/") else model
                print(f"🔧 DEBUG: Mapped to vLLM - {self.litellm_model}")
            else:
                # Assume HuggingFace format
                self.litellm_provider = "huggingface"
                self.litellm_model = f"google/{model}" if not model.startswith("google/") else model
                print(f"🔧 DEBUG: Mapped to HuggingFace - {self.litellm_model}")
        else:
            # Use provider and model as-is for standard litellm providers
            self.litellm_provider = provider
            self.litellm_model = model
            print(f"🔧 DEBUG: Using direct mapping - {provider}/{model}")

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 7
    ) -> SolveResult:
        print(f"\n🚀 DEBUG: Starting solve for task {task_index}")
        
        # Create detailed log file
        log_filename = f"gemma3_debug_detailed_task_{task_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        def log_and_print(message):
            """Helper function to both print and log messages"""
            print(message)
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
        
        log_and_print(f"🚀 DEBUG: Starting solve for task {task_index}")
        log_and_print(f"📅 Timestamp: {datetime.now().isoformat()}")
        log_and_print(f"🔧 Model: {self.model} via {self.provider}")
        log_and_print("=" * 100)
        
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        
        log_and_print(f"🔧 DEBUG: Initial observation: {obs}")
        
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        
        log_and_print(f"🔧 DEBUG: Initial messages constructed:")
        log_and_print(f"   System message length: {len(messages[0]['content'])}")
        log_and_print(f"   User message: {messages[1]['content']}")
        log_and_print(f"   🎯 Task index being run: {task_index}")
        
        # Print the full system message to understand what instructions are being sent
        log_and_print(f"\n📋 FULL SYSTEM MESSAGE (WIKI):")
        log_and_print("=" * 80)
        log_and_print(messages[0]['content'])
        log_and_print("=" * 80)
        
        log_and_print(f"\n📋 FULL USER INSTRUCTION:")
        log_and_print("=" * 80)
        log_and_print(messages[1]['content'])
        log_and_print("=" * 80)
        
        for step in range(max_num_steps):
            log_and_print(f"\n📍 DEBUG: turn {step + 1}/{max_num_steps}")
            log_and_print("🔥" * 60)
            
            # Show the exact parameters being sent
            log_and_print(f"🔧 DEBUG: LiteLLM call parameters:")
            log_and_print(f"   model: {self.litellm_model}")
            log_and_print(f"   custom_llm_provider: {self.litellm_provider}")
            log_and_print(f"   temperature: {self.temperature}")
            log_and_print(f"   tools count: {len(self.tools_info)}")
            log_and_print(f"   messages count: {len(messages)}")
            
            # Log the COMPLETE conversation context being sent to the model
            log_and_print(f"\n💬 COMPLETE CONVERSATION CONTEXT SENT TO MODEL:")
            log_and_print("🔸" * 80)
            for i, msg in enumerate(messages):
                log_and_print(f"MESSAGE {i+1} (Role: {msg['role']}):")
                log_and_print(f"Content: {msg.get('content', 'N/A')}")
                if 'tool_call_id' in msg:
                    log_and_print(f"Tool Call ID: {msg['tool_call_id']}")
                if 'name' in msg:
                    log_and_print(f"Tool Name: {msg['name']}")
                log_and_print("-" * 40)
            log_and_print("🔸" * 80)
            
            # Make the LiteLLM call - only pass tools if we have them
            completion_params = {
                "messages": messages,
                "model": self.litellm_model,
                "custom_llm_provider": self.litellm_provider,
                "temperature": self.temperature,
            }
            
            # Only add tools if we have them
            if self.tools_info:
                completion_params["tools"] = self.tools_info
                log_and_print(f"🔧 DEBUG: Including {len(self.tools_info)} tools in request")
            else:
                log_and_print(f"🔧 DEBUG: No tools included in request")
                
            try:
                res = completion(**completion_params)
                log_and_print(f"✅ DEBUG: LiteLLM response received")
                log_and_print(f"   Model: {res.model}")
                log_and_print(f"   Response cost: {res._hidden_params.get('response_cost', 0.0)}")
                
            except Exception as e:
                log_and_print(f"❌ DEBUG: LiteLLM call failed: {e}")
                break
            
            # Extract and analyze the message
            next_message = res.choices[0].message.model_dump()
            
            log_and_print(f"\n🤖 DEBUG: Raw message from model:")
            log_and_print(f"   Role: {next_message.get('role')}")
            log_and_print(f"   Content: {next_message.get('content')}")
            log_and_print(f"   Tool calls: {next_message.get('tool_calls')}")
            
            # Log the COMPLETE raw response
            log_and_print(f"\n📄 COMPLETE RAW MODEL RESPONSE:")
            log_and_print("🔹" * 80)
            log_and_print(json.dumps(next_message, indent=2, ensure_ascii=False))
            log_and_print("🔹" * 80)
            
            total_cost += res._hidden_params["response_cost"]
            
            # Convert to action with debugging
            action = debug_message_to_action(next_message)
            
            log_and_print(f"🎯 DEBUG: Action created:")
            log_and_print(f"   Name: {action.name}")
            log_and_print(f"   Kwargs: {action.kwargs}")
            
            # Execute the action
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            
            log_and_print(f"🔧 DEBUG: Environment response:")
            log_and_print(f"   Reward: {reward}")
            log_and_print(f"   Done: {env_response.done}")
            log_and_print(f"   Observation: {env_response.observation}")
            
            # Handle conversation continuation
            if action.name != RESPOND_ACTION_NAME:
                log_and_print(f"🔧 DEBUG: Adding tool call to conversation")
                
                # Check if we have actual tool_calls or if we parsed from content
                if next_message.get("tool_calls") is not None and len(next_message["tool_calls"]) > 0:
                    # Standard tool_calls format
                    next_message["tool_calls"] = next_message["tool_calls"][:1]
                    messages.extend(
                        [
                            next_message,
                            {
                                "role": "tool",
                                "tool_call_id": next_message["tool_calls"][0]["id"],
                                "name": next_message["tool_calls"][0]["function"]["name"],
                                "content": env_response.observation,
                            },
                        ]
                    )
                else:
                    # We parsed function call from content - simulate tool_calls structure
                    fake_tool_call = {
                        "id": f"call_{action.name}_{len(messages)}",
                        "type": "function",
                        "function": {
                            "name": action.name,
                            "arguments": json.dumps(action.kwargs)
                        }
                    }
                    next_message["tool_calls"] = [fake_tool_call]
                    messages.extend(
                        [
                            next_message,
                            {
                                "role": "tool",
                                "tool_call_id": fake_tool_call["id"],
                                "name": action.name,
                                "content": env_response.observation,
                            },
                        ]
                    )
            else:
                log_and_print(f"🔧 DEBUG: Adding text response to conversation")
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            
            # Log the updated conversation state
            log_and_print(f"\n📝 CONVERSATION STATE AFTER TURN {step + 1}:")
            log_and_print(f"   Total messages: {len(messages)}")
            log_and_print(f"   Current reward: {reward}")
            log_and_print(f"   Task done: {env_response.done}")
            
            if env_response.done:
                log_and_print(f"🏁 DEBUG: Task completed at step {step + 1}")
                break
                
        log_and_print(f"🎯 DEBUG: Final result - Reward: {reward}, Total cost: {total_cost}")
        log_and_print(f"💾 DEBUG: Complete log saved to: {log_filename}")
        
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )
    


####### cell #######



import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
import os
import json
import random
import traceback
import time
from math import comb
import multiprocessing
from typing import List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from tau_bench.envs import get_env
from tau_bench.agents.base import Agent
from tau_bench.types import EnvRunResult, RunConfig
from tau_bench.run import display_metrics
from litellm import provider_list
from tau_bench.envs.user import UserStrategy


####### cell #######


# Configuration setup based on the specified parameters
@dataclass
class Config:
    agent_strategy: str = "tool-calling"
    env: str = "retail"
    model: str = "gemma3:12b"
    model_provider: str = "gemma"
    user_model: str = "gemini-2.5-pro"
    user_model_provider: str = "vertex_ai"
    user_strategy: str = "llm"
    max_concurrency: int = 1
    task_id: int = 0
    # Additional config parameters that seem to be expected
    user_temperature: float = 0.0
    task_split: str = "test"
    end_index: int = -1
    start_index: int = 0
    task_ids: List[int] = None
    num_trials: int = 1
    shuffle: bool = False

config = Config()

# Set up task_ids if task_id is specified
if config.task_id is not None:
    config.task_ids = [config.task_id]

print(f"🔧 DEBUG: Configuration setup:")
print(f"   task_id: {config.task_id}")
print(f"   task_ids: {config.task_ids}")
print(f"   start_index: {config.start_index}")
print(f"   end_index: {config.end_index}")
print(f"   env: {config.env}")
print(f"   task_split: {config.task_split}")

# Agent factory function
def agent_factory(tools_info, wiki, config):
    return ToolCallingAgent(
        tools_info=tools_info,
        wiki=wiki,
        model=config.model,
        provider=config.model_provider,
        temperature=0.0
    )


####### cell #######


# Checkpoint path setup
ckpt_path = f"gemma3_debug_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Start timing
start_time = time.time()

print(f"Loading user with strategy: {config.user_strategy}")
env = get_env(
    config.env,
    user_strategy=config.user_strategy,
    user_model=config.user_model,
    user_provider=config.user_model_provider,
    user_temperature=config.user_temperature,
    task_split=config.task_split,
)

# Read retail_tools_info.md and append to env.wiki
retail_tools_path = "/home/lixali_google_com/data/adk-lightning/tau-bench-lightning/retail_tools_info.md"
try:
    with open(retail_tools_path, "r") as f:
        retail_tools_content = f.read()
    # Append the tools info to the wiki with a separator
    enhanced_wiki = env.wiki + "\n\n" + "="*80 + "\n" + "AVAILABLE TOOLS DOCUMENTATION:\n" + "="*80 + "\n\n" + retail_tools_content
    print(f"🔧 DEBUG: Enhanced wiki with retail tools info ({len(retail_tools_content)} chars added)")
except FileNotFoundError:
    print(f"⚠️  WARNING: retail_tools_info.md not found at {retail_tools_path}, using original wiki")
    enhanced_wiki = env.wiki

agent = agent_factory(
    # tools_info=None,  # Set to None to disable tools
    wiki=enhanced_wiki,  # Use enhanced wiki with tools info
    config=config,
    tools_info=None,  # Set to None to disable tools
    # tools_info=env.tools_info,  # Uncomment this line to enable tools
)
end_index = (
    len(env.tasks) if config.end_index == -1 else min(config.end_index, len(env.tasks))
)
results: List[EnvRunResult] = []
lock = multiprocessing.Lock()
if config.task_ids and len(config.task_ids) > 0:
    print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
else:
    print(
        f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
)
for i in range(config.num_trials):
    if config.task_ids and len(config.task_ids) > 0:
        idxs = config.task_ids
    else:
        idxs = list(range(config.start_index, end_index))
    if config.shuffle:
        random.shuffle(idxs)

    def _run(idx: int) -> EnvRunResult:
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
                info={"error": str(e), "traceback": traceback.format_exc()},
                traj=[],
                trial=i,
            )
        print(
            "✅" if result.reward == 1 else "❌",
            f"task_id={idx}",
            result.info,
        )
        print("-----")
        with lock:
            data = []
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    data = json.load(f)
            with open(ckpt_path, "w") as f:
                json.dump(data + [result.model_dump()], f, indent=2)
        return result

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as executor:
        res = list(executor.map(_run, idxs))
        results.extend(res)

metrics = display_metrics(results)

# Calculate total runtime
end_time = time.time()
total_runtime = end_time - start_time
runtime_minutes = total_runtime / 60
runtime_hours = total_runtime / 3600

# Print runtime to terminal
print(f"\n⏱️  Total Runtime: {total_runtime:.2f} seconds ({runtime_minutes:.2f} minutes, {runtime_hours:.2f} hours)")