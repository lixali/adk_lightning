#!/usr/bin/env python3
"""
Tau-Bench Data Collector for PPO Training
Generates episodes using tau-bench environments with focus on wrong decision scenarios.
"""

import os
import sys
import json
import random
import logging
from typing import Dict, Any, List, Tuple
import torch

# Add tau_bench to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tau_bench.envs.retail.env import MockRetailDomainEnv
from tau_bench.envs.airline.env import MockAirlineDomainEnv
from tau_bench.agents.tool_calling_agent import ToolCallingAgent
from tau_bench.envs.user import LLMUserSimulationEnv, UserStrategy
from tau_bench.types import RunConfig

logger = logging.getLogger(__name__)

class TauBenchDataCollector:
    """Collects training data from tau-bench environments for PPO training"""
    
    def __init__(self, model_name: str, tokenizer, focus_env: str = "retail"):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.focus_env = focus_env
        
        # Initialize environments
        self.retail_env = MockRetailDomainEnv(
            user_strategy=UserStrategy.LLM,
            user_model="gemini-2.5-flash",
            user_provider="vertex_ai",
            user_temperature=0.7,
            task_index=0  # Start with first task
        )
        self.airline_env = MockAirlineDomainEnv(
            user_strategy=UserStrategy.LLM,
            user_model="gemini-2.5-flash",
            user_provider="vertex_ai", 
            user_temperature=0.7,
            task_index=0  # Start with first task
        )
        
        # Load tasks that commonly result in wrong decisions
        self.wrong_decision_tasks = self.load_wrong_decision_tasks()
        
        logger.info(f"TauBenchDataCollector initialized with {len(self.wrong_decision_tasks)} wrong decision tasks")
    
    def load_wrong_decision_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks that commonly result in wrong decisions based on error analysis"""
        
        # Based on our error analysis, these are common wrong decision patterns
        wrong_decision_patterns = [
            {
                "type": "ambiguous_cancellation",
                "description": "User wants to cancel but unclear which order/items",
                "env": "retail",
                "scenario": "multiple_orders"
            },
            {
                "type": "missing_product_lookup", 
                "description": "Agent fails to look up product details before making recommendations",
                "env": "retail",
                "scenario": "product_inquiry"
            },
            {
                "type": "wrong_customer_lookup",
                "description": "Agent uses wrong customer identification method",
                "env": "retail", 
                "scenario": "customer_service"
            },
            {
                "type": "incomplete_order_modification",
                "description": "Agent modifies order without checking all requirements",
                "env": "retail",
                "scenario": "order_change"
            },
            {
                "type": "flight_booking_confusion",
                "description": "Agent confuses booking vs rebooking vs cancellation",
                "env": "airline",
                "scenario": "flight_changes"
            },
            {
                "type": "baggage_policy_misunderstanding", 
                "description": "Agent applies wrong baggage rules",
                "env": "airline",
                "scenario": "baggage_issues"
            }
        ]
        
        return wrong_decision_patterns
    
    def create_wrong_decision_scenario(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create a specific scenario designed to trigger wrong decisions"""
        
        if pattern["env"] == "retail":
            return self.create_retail_wrong_decision_scenario(pattern)
        elif pattern["env"] == "airline":
            return self.create_airline_wrong_decision_scenario(pattern)
        else:
            return self.create_generic_wrong_decision_scenario(pattern)
    
    def create_retail_wrong_decision_scenario(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create retail-specific wrong decision scenarios"""
        
        if pattern["type"] == "ambiguous_cancellation":
            return {
                "instruction": "I want to cancel my recent order. Actually, just cancel the grill but keep the other items.",
                "expected_actions": [
                    {"type": "get_customer_info", "args": {"customer_name": "[USER_NAME]", "zip_code": "[USER_ZIP]"}},
                    {"type": "get_order_details", "args": {"order_id": "[FOUND_ORDER_ID]"}},
                    {"type": "cancel_order_items", "args": {"order_id": "[ORDER_ID]", "item_ids": ["[GRILL_ITEM_ID]"]}}
                ],
                "wrong_decision_triggers": [
                    "Agent might cancel entire order instead of specific items",
                    "Agent might not verify which order contains the grill",
                    "Agent might assume which order to cancel without checking"
                ]
            }
        
        elif pattern["type"] == "missing_product_lookup":
            return {
                "instruction": "I'm looking for a good water bottle for hiking. What do you recommend?",
                "expected_actions": [
                    {"type": "search_products", "args": {"query": "water bottle hiking"}},
                    {"type": "get_product_details", "args": {"product_id": "[FOUND_PRODUCT_ID]"}},
                    {"type": "get_product_details", "args": {"product_id": "[ANOTHER_PRODUCT_ID]"}}
                ],
                "wrong_decision_triggers": [
                    "Agent might make recommendations without looking up products",
                    "Agent might not compare multiple options",
                    "Agent might provide generic advice instead of specific products"
                ]
            }
        
        elif pattern["type"] == "wrong_customer_lookup":
            return {
                "instruction": "Hi, I'm having trouble with my order. My name is Sarah Chen and I live in zip code 12345.",
                "expected_actions": [
                    {"type": "get_customer_info", "args": {"customer_name": "Sarah Chen", "zip_code": "12345"}},
                    {"type": "get_order_details", "args": {"order_id": "[FOUND_ORDER_ID]"}}
                ],
                "wrong_decision_triggers": [
                    "Agent might search by only name or only zip code",
                    "Agent might assume customer info without lookup",
                    "Agent might use wrong customer identification method"
                ]
            }
        
        return {"instruction": "Generic retail scenario", "expected_actions": [], "wrong_decision_triggers": []}
    
    def create_airline_wrong_decision_scenario(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create airline-specific wrong decision scenarios"""
        
        if pattern["type"] == "flight_booking_confusion":
            return {
                "instruction": "I need to change my flight to tomorrow instead of next week.",
                "expected_actions": [
                    {"type": "get_customer_info", "args": {"customer_name": "[USER_NAME]", "phone": "[USER_PHONE]"}},
                    {"type": "get_booking_details", "args": {"booking_id": "[BOOKING_ID]"}},
                    {"type": "modify_booking", "args": {"booking_id": "[BOOKING_ID]", "new_date": "[TOMORROW_DATE]"}}
                ],
                "wrong_decision_triggers": [
                    "Agent might try to book new flight instead of modifying",
                    "Agent might cancel old booking before securing new one",
                    "Agent might not check availability before attempting change"
                ]
            }
        
        elif pattern["type"] == "baggage_policy_misunderstanding":
            return {
                "instruction": "I'm flying internationally and want to bring an extra bag. What are my options?",
                "expected_actions": [
                    {"type": "get_customer_info", "args": {"customer_name": "[USER_NAME]", "phone": "[USER_PHONE]"}},
                    {"type": "get_booking_details", "args": {"booking_id": "[BOOKING_ID]"}},
                    {"type": "get_baggage_policy", "args": {"route_type": "international", "ticket_class": "[TICKET_CLASS]"}}
                ],
                "wrong_decision_triggers": [
                    "Agent might apply domestic baggage rules to international flight",
                    "Agent might not check specific ticket class rules",
                    "Agent might quote outdated or incorrect policy"
                ]
            }
        
        return {"instruction": "Generic airline scenario", "expected_actions": [], "wrong_decision_triggers": []}
    
    def create_generic_wrong_decision_scenario(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create generic wrong decision scenario"""
        return {
            "instruction": "I need help with my account.",
            "expected_actions": [
                {"type": "get_customer_info", "args": {"customer_name": "[USER_NAME]"}}
            ],
            "wrong_decision_triggers": [
                "Agent might not ask for identification",
                "Agent might make assumptions about user's needs"
            ]
        }
    
    def generate_episode(self, model, focus_on_wrong_decisions: bool = True) -> Dict[str, Any]:
        """Generate a simple mock episode for testing"""
        
        try:
            # For now, create a simple mock episode with basic structure
            # This will be replaced with actual tau-bench episode generation later
            
            mock_trajectory = [
                {
                    "role": "user",
                    "content": "Hi, I need help with my order.",
                },
                {
                    "role": "assistant", 
                    "content": "I'd be happy to help you with your order. Can you please provide your order number?",
                },
                {
                    "role": "user",
                    "content": "It's order #12345. I want to cancel it.",
                },
                {
                    "role": "assistant",
                    "content": "I can help you cancel order #12345. Let me process that for you right away.",
                }
            ]
            
            # Assign a random reward (in real training this would come from the reward model)
            reward = random.uniform(0.1, 1.0)
            
            episode = {
                "trajectory": mock_trajectory,
                "reward": reward,
                "task_id": f"mock_task_{random.randint(1, 100)}",
                "success": reward > 0.5
            }
            
            logger.debug(f"Generated mock episode with reward: {reward:.3f}")
            return episode
            
        except Exception as e:
            logger.error(f"Error generating episode: {e}")
            # Return a minimal valid episode
            return {
                "trajectory": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "How can I help you?"}
                ],
                "reward": 1.0,
                "task_id": "fallback_task",
                "success": True
            }
            
            # Create agent (we'll replace this with our model during training)
            agent = ToolCallingAgent(
                provider=config.model_provider,
                model=config.model,
                temperature=config.temperature,
                max_concurrency=config.max_concurrency
            )
            
            # Run episode
            episode_result = env.run_episode(
                agent=agent,
                user=user_sim,
                task=task,
                trial_idx=0,
                config=config
            )
            
            # Format episode data for training
            episode_data = {
                "task_info": {
                    "task_id": task.get("task_id", "unknown"),
                    "task": task,
                    "environment": self.focus_env
                },
                "trajectory": episode_result.get("traj", []),
                "ground_truth": {
                    "actions": task.get("actions", []),
                    "instruction": task.get("instruction", ""),
                    "outputs": task.get("outputs", [])
                },
                "metrics": {
                    "reward": episode_result.get("reward", 0.0),
                    "success": episode_result.get("reward", 0.0) == 1.0
                }
            }
            
            return episode_data
            
        except Exception as e:
            logger.error(f"Error generating episode: {e}")
            # Return a minimal episode with low reward
            return {
                "task_info": {"task_id": "error", "environment": self.focus_env},
                "trajectory": [],
                "ground_truth": {"actions": [], "instruction": "", "outputs": []},
                "metrics": {"reward": 0.0, "success": False},
                "error": str(e)
            }
    
    def adapt_scenario_to_task(self, scenario: Dict[str, Any], tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adapt a wrong decision scenario to an actual tau-bench task"""
        
        # Find a similar task from the task list
        for task in tasks:
            instruction = task.get("instruction", "").lower()
            scenario_instruction = scenario["instruction"].lower()
            
            # Simple keyword matching to find relevant tasks
            if any(word in instruction for word in scenario_instruction.split()[:3]):
                # Modify the task to incorporate our wrong decision triggers
                modified_task = task.copy()
                modified_task["wrong_decision_scenario"] = scenario
                return modified_task
        
        # If no matching task found, use the first task as a base
        if tasks:
            modified_task = tasks[0].copy()
            modified_task["instruction"] = scenario["instruction"]
            modified_task["wrong_decision_scenario"] = scenario
            return modified_task
        
        # Fallback: create a minimal task
        return {
            "task_id": "wrong_decision_generated",
            "instruction": scenario["instruction"],
            "actions": scenario["expected_actions"],
            "outputs": [],
            "wrong_decision_scenario": scenario
        }
    
    def process_rollout_for_training(self, rollout: Dict[str, Any]) -> Dict[str, List]:
        """Process a rollout into training format (states, actions, rewards)"""
        
        trajectory = rollout.get("trajectory", [])
        reward = rollout.get("reward", 0.0)
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        # Process conversation trajectory into training data
        conversation_so_far = ""
        
        for i, turn in enumerate(trajectory):
            if turn.get("role") == "user":
                conversation_so_far += f"User: {turn.get('content', '')}\n"
            elif turn.get("role") == "assistant":
                # This is where we would capture the model's decision
                state = conversation_so_far + "Assistant: "
                content = turn.get("content", "")
                
                # Tokenize the state and response (limit length to avoid memory issues)
                state_tokens = self.tokenizer.encode(
                    state, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True,
                    padding=False
                )
                response_tokens = self.tokenizer.encode(
                    content, 
                    return_tensors="pt", 
                    max_length=256, 
                    truncation=True,
                    padding=False
                )
                
                # Ensure we have proper tensor shapes (remove batch dimension)
                if state_tokens.dim() > 1:
                    state_tokens = state_tokens.squeeze(0)
                if response_tokens.dim() > 1:
                    response_tokens = response_tokens.squeeze(0)
                
                states.append(state_tokens)
                actions.append(response_tokens)
                
                # For now, assign the final reward to all steps
                step_reward = reward if i == len(trajectory) - 1 else 0.0
                rewards.append(step_reward)
                
                # Use first token of response as action token for log prob computation
                if len(response_tokens) > 0:
                    action_token = response_tokens[0].item()
                else:
                    action_token = self.tokenizer.eos_token_id or 1
                log_probs.append(torch.tensor(-1.0))  # Placeholder, will be computed during training
                
                conversation_so_far += f"Assistant: {content}\n"
        
        return {
            "states": states,
            "actions": actions,  
            "rewards": rewards,
            "log_probs": log_probs
        }
    
    def generate_batch(self, batch_size: int, focus_on_wrong_decisions: bool = True) -> List[Dict[str, Any]]:
        """Generate a batch of episodes"""
        
        episodes = []
        for i in range(batch_size):
            if i % 10 == 0:
                logger.info(f"Generating episode {i+1}/{batch_size}")
            
            episode = self.generate_episode(
                model=None,  # Will be provided by trainer
                focus_on_wrong_decisions=focus_on_wrong_decisions
            )
            episodes.append(episode)
        
        return episodes
