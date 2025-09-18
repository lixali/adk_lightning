import json
from litellm import completion
import litellm
import sys
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
import argparse
from dataclasses import dataclass

from tau_bench.agents.base import Agent
from tau_bench.agents import debug_message_to_action
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


def convert_messages_for_gemma(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert OpenAI-style messages to Gemma-compatible format.
    Gemma only supports 'user' role, so we convert all messages to user messages
    with clear role indicators.
    """
    gemma_messages = []
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if role == "system":
            # Convert system message to user message with clear system indicator
            new_content = f"# System Instructions\n\n{content}"
            gemma_messages.append({"role": "user", "content": new_content})
            
        elif role == "assistant":
            # Convert assistant message to user message with assistant indicator
            if msg.get("tool_calls"):
                # Assistant made a tool call
                tool_call = msg["tool_calls"][0]  # Take first tool call
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]
                new_content = f"# Assistant Tool Call\n\nFunction: {func_name}\nArguments: {func_args}"
            else:
                # Regular assistant response
                new_content = f"# Assistant Response\n\n{content}"
            gemma_messages.append({"role": "user", "content": new_content})
            
        elif role == "tool":
            # Convert tool result to user message with tool indicator
            tool_name = msg.get("name", "unknown")
            new_content = f"# Tool Result ({tool_name})\n\n{content}"
            gemma_messages.append({"role": "user", "content": new_content})
            
        else:
            # Regular user message - keep as is but ensure role is 'user'
            gemma_messages.append({"role": "user", "content": content})
    
    return gemma_messages


class VLLMToolCallingAgent(Agent):
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
        
        print(f"🔧 DEBUG: VLLM Agent initialized with:")
        print(f"   Model: {model}")
        print(f"   Provider: {provider}")
        print(f"   Temperature: {temperature}")
        print(f"   Wiki length: {len(wiki)}")
        print(f"   Tools count: {len(tools_info) if tools_info else 0} -> LIMITED TO: {len(self.tools_info)}")
        if self.tools_info:
            print(f"   Available tools: {[tool['function']['name'] for tool in self.tools_info]}")
        else:
            print(f"   Available tools: None (no tools provided)")
        
        # Handle vLLM provider mapping
        if provider == "vllm":
            # Use openai-compatible API instead of local vLLM to avoid torch compatibility issues
            self.litellm_provider = "openai"
            # Point to your vLLM server endpoint (adjust as needed)
            vllm_base_url = "http://localhost:8001/v1"  # Default vLLM OpenAI-compatible endpoint
            
            # Set the model name and base URL for OpenAI-compatible vLLM server
            if model.startswith("vllm/"):
                model_name = model.replace("vllm/", "")
            else:
                model_name = model
            
            self.litellm_model = model_name
            self.vllm_base_url = vllm_base_url
            
            print(f"🔧 DEBUG: Using vLLM via OpenAI-compatible API:")
            print(f"   Model: {self.litellm_model}")
            print(f"   Base URL: {self.vllm_base_url}")
            print(f"   Provider: {self.litellm_provider}")
        else:
            # Use provider and model as-is for standard litellm providers
            self.litellm_provider = provider
            self.litellm_model = model
            self.vllm_base_url = None
            print(f"🔧 DEBUG: Using direct mapping - {provider}/{model}")

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        print(f"\n🚀 DEBUG: Starting VLLM solve for task {task_index}")
        
        # Create detailed log file
        log_filename = f"vllm_gemma3_debug_detailed_task_{task_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        def log_and_print(message):
            """Helper function to both print and log messages"""
            print(message)
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
        
        log_and_print(f"🚀 DEBUG: Starting VLLM solve for task {task_index}")
        log_and_print(f"📅 Timestamp: {datetime.now().isoformat()}")
        log_and_print(f"🔧 Model: {self.model} via {self.provider} (vLLM)")
        log_and_print("=" * 100)
        
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        
        log_and_print(f"🔧 DEBUG: Initial observation: {obs}")
        
        # Create standard OpenAI format messages first
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
            log_and_print(f"🔧 DEBUG: LiteLLM vLLM call parameters:")
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
            # Convert messages to Gemma-compatible format (only 'user' role supported)
            gemma_messages = convert_messages_for_gemma(messages)
            
            # 🔧 TRUNCATE TO ~100 TOKENS FOR TESTING
            def truncate_messages_to_approx_tokens(messages, target_tokens=100):
                """Truncate messages to approximately target_tokens (rough word-based estimation)"""
                # Rough estimation: 1 token ≈ 0.75 words, so 100 tokens ≈ 75 words
                target_words = int(target_tokens * 0.75)
                
                truncated_messages = []
                word_count = 0
                
                for msg in messages:
                    content = msg.get("content", "")
                    words = content.split()
                    
                    if word_count + len(words) <= target_words:
                        # Full message fits
                        truncated_messages.append(msg)
                        word_count += len(words)
                    else:
                        # Truncate this message and stop
                        remaining_words = target_words - word_count
                        if remaining_words > 10:  # Only add if substantial content remains
                            truncated_content = " ".join(words[:remaining_words]) + "... [TRUNCATED FOR TESTING]"
                            truncated_msg = msg.copy()
                            truncated_msg["content"] = truncated_content
                            truncated_messages.append(truncated_msg)
                        break
                
                log_and_print(f"🔧 DEBUG: Truncated from {sum(len(m.get('content','').split()) for m in messages)} words to ~{word_count} words (~{int(word_count/0.75)} tokens)")
                return truncated_messages
            
            # Apply truncation for testing
            gemma_messages = truncate_messages_to_approx_tokens(gemma_messages, target_tokens=100)
            
            completion_params = {
                "messages": gemma_messages,
                "model": self.litellm_model,
                "temperature": self.temperature,
            }
            
            # Add base URL for vLLM OpenAI-compatible server
            if hasattr(self, 'vllm_base_url') and self.vllm_base_url:
                completion_params["api_base"] = self.vllm_base_url
                completion_params["api_key"] = "EMPTY"  # vLLM doesn't require API key
                completion_params["custom_llm_provider"] = "openai"  # Specify OpenAI-compatible provider
                log_and_print(f"🔧 DEBUG: Using vLLM server at {self.vllm_base_url}")
            else:
                completion_params["custom_llm_provider"] = self.litellm_provider
            
            # Only add tools if we have them - NOTE: Gemma may not support tools properly
            if self.tools_info:
                completion_params["tools"] = self.tools_info
                log_and_print(f"🔧 DEBUG: Including {len(self.tools_info)} tools in request")
            else:
                log_and_print(f"🔧 DEBUG: No tools included in request")
            
            # Log the converted messages for Gemma
            log_and_print(f"\n💬 GEMMA-CONVERTED MESSAGES:")
            log_and_print("🔹" * 80)
            for i, msg in enumerate(gemma_messages):
                log_and_print(f"GEMMA MESSAGE {i+1} (Role: {msg['role']}):")
                log_and_print(f"Content: {msg.get('content', 'N/A')[:500]}...")  # Truncate for readability
                log_and_print("-" * 40)
            log_and_print("🔹" * 80)
            
            # Log the exact completion parameters
            log_and_print(f"\n🎯 EXACT COMPLETION PARAMETERS:")
            log_and_print("🟡" * 80)
            log_and_print(json.dumps(completion_params, indent=2, ensure_ascii=False, default=str))
            log_and_print("🟡" * 80)
                
            try:
                log_and_print(f"\n🚀 CALLING vLLM via LiteLLM...")
                
                # Enable debug mode to capture the raw request
                import litellm
                import requests
                import json as json_lib
                
                # Capture the actual HTTP request that will be sent to vLLM
                log_and_print(f"\n🔍 INTERCEPTING RAW HTTP REQUEST TO vLLM...")
                
                # Make a direct request to get the exact prompt format
                vllm_url = completion_params.get("api_base", "http://localhost:8001/v1")
                if not vllm_url.endswith("/"):
                    vllm_url += "/"
                
                # Prepare the request payload exactly as LiteLLM would send it
                raw_payload = {
                    "model": completion_params["model"],
                    "messages": completion_params["messages"],
                    "temperature": completion_params.get("temperature", 0.0),
                    "max_tokens": completion_params.get("max_tokens", 1000),
                    "stream": False
                }
                
                log_and_print(f"📡 RAW HTTP REQUEST PAYLOAD TO vLLM:")
                log_and_print("🔴" * 80)
                log_and_print(json_lib.dumps(raw_payload, indent=2, ensure_ascii=False))
                log_and_print("🔴" * 80)
                
                # Try to get the tokenized prompt from vLLM debug endpoint if available
                try:
                    debug_url = vllm_url.replace("/v1/", "/debug/")
                    log_and_print(f"🔧 Attempting to get tokenized prompt from debug endpoint: {debug_url}")
                except:
                    pass
                
                # Make a special request to get the formatted prompt
                try:
                    # Some vLLM servers support getting the formatted prompt by adding a special parameter
                    prompt_debug_payload = raw_payload.copy()
                    prompt_debug_payload["echo"] = True  # This might return the formatted prompt
                    prompt_debug_payload["max_tokens"] = 1  # Minimal generation
                    
                    headers = {"Content-Type": "application/json"}
                    
                    log_and_print(f"🔍 MAKING DEBUG REQUEST TO GET FORMATTED PROMPT...")
                    debug_response = requests.post(
                        vllm_url + "chat/completions",
                        json=prompt_debug_payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    if debug_response.status_code == 200:
                        debug_result = debug_response.json()
                        log_and_print(f"📝 FORMATTED PROMPT FROM vLLM:")
                        log_and_print("🟢" * 80)
                        
                        # Look for the formatted prompt in various places
                        if "choices" in debug_result and len(debug_result["choices"]) > 0:
                            choice = debug_result["choices"][0]
                            if "text" in choice:
                                log_and_print(f"FORMATTED PROMPT (from 'text' field):")
                                log_and_print(choice["text"])
                            elif "message" in choice and "content" in choice["message"]:
                                log_and_print(f"FORMATTED PROMPT (from message content):")
                                log_and_print(choice["message"]["content"])
                        
                        # Also check if there's a prompt field in the response
                        if "prompt" in debug_result:
                            log_and_print(f"FORMATTED PROMPT (from 'prompt' field):")
                            log_and_print(debug_result["prompt"])
                            
                        log_and_print("🟢" * 80)
                    else:
                        log_and_print(f"❌ Debug request failed: {debug_response.status_code}")
                        
                except Exception as debug_e:
                    log_and_print(f"⚠️ Debug request failed: {debug_e}")
                
                # Alternative approach: Try to use the transformers tokenizer to format the prompt
                try:
                    log_and_print(f"\n🔧 ALTERNATIVE: Trying to format prompt using tokenizer...")
                    from transformers import AutoTokenizer
                    
                    # Try to load the tokenizer for the model
                    model_name = "google/gemma-3-12b-it"  # Adjust if needed
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    
                    if hasattr(tokenizer, 'apply_chat_template'):
                        formatted_prompt = tokenizer.apply_chat_template(
                            completion_params["messages"], 
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        log_and_print(f"📝 FORMATTED PROMPT (via transformers tokenizer):")
                        log_and_print("🟠" * 80)
                        log_and_print(formatted_prompt)
                        log_and_print("🟠" * 80)
                    else:
                        log_and_print(f"⚠️ Tokenizer doesn't have apply_chat_template method")
                        
                except Exception as tokenizer_e:
                    log_and_print(f"⚠️ Tokenizer formatting failed: {tokenizer_e}")
                
                res = completion(**completion_params)
                log_and_print(f"✅ DEBUG: LiteLLM vLLM response received")
                log_and_print(f"   Model: {res.model}")
                log_and_print(f"   Response cost: {res._hidden_params.get('response_cost', 0.0)}")
                
            except Exception as e:
                log_and_print(f"❌ DEBUG: LiteLLM vLLM call failed: {e}")
                import traceback
                log_and_print(f"Full traceback: {traceback.format_exc()}")
                break
            
            # Extract and analyze the message
            next_message = res.choices[0].message.model_dump()
            
            log_and_print(f"\n🤖 DEBUG: Raw message from vLLM model:")
            log_and_print(f"   Role: {next_message.get('role')}")
            log_and_print(f"   Content: {next_message.get('content')}")
            log_and_print(f"   Tool calls: {next_message.get('tool_calls')}")
            
            # Log the COMPLETE raw response
            log_and_print(f"\n📄 COMPLETE RAW vLLM MODEL RESPONSE:")
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
                
                # CRITICAL FIX: After tool execution, agent must process result and respond to user
                log_and_print(f"🔧 DEBUG: Tool executed, now getting agent response to tool result...")
                
                # Make another LLM call for agent to process tool result and respond to user
                # Convert messages to Gemma-compatible format
                gemma_follow_up_messages = convert_messages_for_gemma(messages)
                
                completion_params = {
                    "messages": gemma_follow_up_messages,
                    "model": self.litellm_model,
                    "temperature": self.temperature,
                }
                
                # Add base URL for vLLM OpenAI-compatible server
                if hasattr(self, 'vllm_base_url') and self.vllm_base_url:
                    completion_params["api_base"] = self.vllm_base_url
                    completion_params["api_key"] = "EMPTY"
                else:
                    completion_params["custom_llm_provider"] = self.litellm_provider
                
                # Only add tools if we have them
                if self.tools_info:
                    completion_params["tools"] = self.tools_info
                    
                try:
                    follow_up_res = completion(**completion_params)
                    log_and_print(f"✅ DEBUG: Follow-up LiteLLM vLLM response received")
                    total_cost += follow_up_res._hidden_params["response_cost"]
                    
                    # Extract the follow-up message
                    follow_up_message = follow_up_res.choices[0].message.model_dump()
                    
                    log_and_print(f"🤖 DEBUG: Follow-up message from vLLM model:")
                    log_and_print(f"   Role: {follow_up_message.get('role')}")
                    log_and_print(f"   Content: {follow_up_message.get('content')}")
                    log_and_print(f"   Tool calls: {follow_up_message.get('tool_calls')}")
                    
                    # Convert follow-up to action
                    follow_up_action = debug_message_to_action(follow_up_message)
                    
                    log_and_print(f"🎯 DEBUG: Follow-up action created:")
                    log_and_print(f"   Name: {follow_up_action.name}")
                    log_and_print(f"   Kwargs: {follow_up_action.kwargs}")
                    
                    # Execute the follow-up action (should usually be a text response)
                    follow_up_env_response = env.step(follow_up_action)
                    reward = follow_up_env_response.reward
                    info = {**info, **follow_up_env_response.info.model_dump()}
                    
                    log_and_print(f"🔧 DEBUG: Follow-up environment response:")
                    log_and_print(f"   Reward: {reward}")
                    log_and_print(f"   Done: {follow_up_env_response.done}")
                    log_and_print(f"   Observation: {follow_up_env_response.observation}")
                    
                    # Add the follow-up response to conversation
                    if follow_up_action.name != RESPOND_ACTION_NAME:
                        # If somehow agent made another tool call, handle it
                        log_and_print(f"🔧 DEBUG: Adding follow-up tool call to conversation")
                        if follow_up_message.get("tool_calls") is not None and len(follow_up_message["tool_calls"]) > 0:
                            follow_up_message["tool_calls"] = follow_up_message["tool_calls"][:1]
                            messages.extend(
                                [
                                    follow_up_message,
                                    {
                                        "role": "tool",
                                        "tool_call_id": follow_up_message["tool_calls"][0]["id"],
                                        "name": follow_up_message["tool_calls"][0]["function"]["name"],
                                        "content": follow_up_env_response.observation,
                                    },
                                ]
                            )
                        else:
                            fake_follow_up_tool_call = {
                                "id": f"call_{follow_up_action.name}_{len(messages)}",
                                "type": "function",
                                "function": {
                                    "name": follow_up_action.name,
                                    "arguments": json.dumps(follow_up_action.kwargs)
                                }
                            }
                            follow_up_message["tool_calls"] = [fake_follow_up_tool_call]
                            messages.extend(
                                [
                                    follow_up_message,
                                    {
                                        "role": "tool",
                                        "tool_call_id": fake_follow_up_tool_call["id"],
                                        "name": follow_up_action.name,
                                        "content": follow_up_env_response.observation,
                                    },
                                ]
                            )
                    else:
                        # Normal case: agent provided text response after tool execution
                        log_and_print(f"🔧 DEBUG: Adding follow-up text response to conversation")
                        messages.extend(
                            [
                                follow_up_message,
                                {"role": "user", "content": follow_up_env_response.observation},
                            ]
                        )
                    
                    # Update completion status
                    if follow_up_env_response.done:
                        log_and_print(f"🏁 DEBUG: Task completed after follow-up at step {step + 1}")
                        break
                        
                except Exception as e:
                    log_and_print(f"❌ DEBUG: Follow-up LiteLLM vLLM call failed: {e}")
                    break
                    
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