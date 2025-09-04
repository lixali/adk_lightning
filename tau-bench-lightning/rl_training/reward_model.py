#!/usr/bin/env python3
"""
Gemini-2.5-pro Reward Model for PPO Training
Uses Vertex AI Gemini to provide rewards based on task success and decision quality.
"""

import os
import json
import logging
from typing import Dict, Any, List
import vertexai
from vertexai.preview.generative_models import GenerativeModel

logger = logging.getLogger(__name__)

class GeminiRewardModel:
    """Reward model using Gemini-2.5-pro to evaluate agent performance"""
    
    def __init__(self, project_id: str = None, location: str = "us-central1"):
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.model = None
        self.setup_model()
    
    def setup_model(self):
        """Initialize Vertex AI and Gemini model"""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel("gemini-2.5-pro")
            logger.info("Gemini-2.5-pro reward model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def create_reward_prompt(self, episode_data: Dict[str, Any]) -> str:
        """Create a comprehensive prompt for reward calculation"""
        
        # Extract episode information
        task_info = episode_data.get("task_info", {})
        trajectory = episode_data.get("trajectory", [])
        ground_truth = episode_data.get("ground_truth", {})
        
        # Get ground truth actions and expected outcomes
        gt_actions = ground_truth.get("actions", [])
        gt_instruction = ground_truth.get("instruction", "")
        gt_outputs = ground_truth.get("outputs", [])
        
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
        
        # Create the reward prompt with one-shot example
        prompt = f"""
You are an expert reward model for training AI assistants in customer service scenarios. Your job is to evaluate how well the agent handled the task and provide a reward score between 0.0 and 1.0.

**EVALUATION CRITERIA:**
1. **Correct Decision Making (40%)** - Did the agent make the right type of tool calls based on the user request?
2. **Accurate Information (30%)** - Did the agent provide correct information and use tool arguments properly?
3. **Task Completion (20%)** - Did the agent successfully complete the user's request?
4. **User Experience (10%)** - Was the interaction smooth and professional?

**ONE-SHOT EXAMPLE OF EXCELLENT PERFORMANCE (Reward: 1.0):**
User Request: "I'm Sofia Martinez from 90210. I want to cancel my recent grill order but keep everything else."

Expected Ground Truth Actions:
- get_customer_info(customer_name="Sofia Martinez", zip_code="90210")
- get_order_details for found orders
- cancel_order for the grill order specifically

EXCELLENT Agent Response:
1. ASSISTANT: "I'll help you cancel your grill order. Let me first look up your account."
2. ASSISTANT (tool calls): [{{"name": "get_customer_info", "arguments": {{"customer_name": "Sofia Martinez", "zip_code": "90210"}}}}]
3. USER: Returns customer info with order IDs
4. ASSISTANT (tool calls): [{{"name": "get_order_details", "arguments": {{"order_id": "ABC123"}}}}]
5. USER: Shows order contains grill + other items
6. ASSISTANT (tool calls): [{{"name": "cancel_order", "arguments": {{"order_id": "ABC123", "reason": "Customer request"}}}}]
7. ASSISTANT: "Perfect! I've cancelled your grill order ABC123. Your other orders remain active."

Why this gets 1.0: Perfect decision making (correct tool sequence), accurate use of provided information, complete task resolution, excellent user communication.

**NOW EVALUATE THIS TASK:**

**User Instruction:** {gt_instruction}

**Expected Ground Truth Actions:** {json.dumps(gt_actions, indent=2)}

**Expected Outputs:** {gt_outputs}

**Agent's Actual Conversation:**
{conversation_text}

**REWARD CALCULATION GUIDELINES:**

**Decision Making Quality (0.0-0.4):**
- 0.4: Perfect tool selection and sequence
- 0.3: Mostly correct tools with minor inefficiencies  
- 0.2: Some correct tools but missing key actions
- 0.1: Wrong tools chosen but attempting to help
- 0.0: Completely wrong approach or no tools used

**Information Accuracy (0.0-0.3):**
- 0.3: All tool arguments correct, no hallucinated info
- 0.2: Minor argument errors but mostly accurate
- 0.1: Some correct info but notable errors
- 0.0: Major inaccuracies or complete hallucination

**Task Completion (0.0-0.2):**
- 0.2: Task fully completed as requested
- 0.1: Partially completed or incomplete resolution
- 0.0: Task not completed

**User Experience (0.0-0.1):**
- 0.1: Professional, clear communication
- 0.05: Adequate communication
- 0.0: Poor or confusing communication

Please provide your evaluation in this exact format:

DECISION_MAKING_SCORE: [0.0-0.4]
INFORMATION_ACCURACY_SCORE: [0.0-0.3] 
TASK_COMPLETION_SCORE: [0.0-0.2]
USER_EXPERIENCE_SCORE: [0.0-0.1]
TOTAL_REWARD: [sum of above scores, 0.0-1.0]
REASONING: [Brief explanation of your scoring]

Focus especially on identifying "wrong decision" scenarios where the agent fails to understand domain rules or makes incorrect tool choices, as this is the most common error type we want to improve.
"""
        return prompt
    
    def calculate_reward(self, episode_data: Dict[str, Any]) -> float:
        """Calculate reward using Gemini-2.5-pro"""
        try:
            prompt = self.create_reward_prompt(episode_data)
            
            # Generate response from Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse the response to extract reward
            reward = self.parse_reward_response(response_text)
            
            # Store detailed evaluation for analysis
            episode_data['reward_details'] = {
                'gemini_response': response_text,
                'parsed_reward': reward
            }
            
            return reward
            
        except Exception as e:
            logger.error(f"Error calculating reward with Gemini: {e}")
            # Return a default low reward for failed evaluation
            return 0.1
    
    def parse_reward_response(self, response_text: str) -> float:
        """Parse Gemini response to extract numerical reward"""
        try:
            lines = response_text.split('\n')
            total_reward = 0.0
            
            for line in lines:
                if line.startswith("TOTAL_REWARD:"):
                    reward_str = line.split(":", 1)[1].strip()
                    # Extract numerical value
                    import re
                    numbers = re.findall(r'\d+\.?\d*', reward_str)
                    if numbers:
                        total_reward = float(numbers[0])
                        break
            
            # Ensure reward is in valid range
            total_reward = max(0.0, min(1.0, total_reward))
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Error parsing reward response: {e}")
            logger.error(f"Response text: {response_text}")
            # Return moderate reward for parsing errors
            return 0.3
    
    def batch_calculate_rewards(self, episodes: List[Dict[str, Any]]) -> List[float]:
        """Calculate rewards for a batch of episodes"""
        rewards = []
        
        for i, episode in enumerate(episodes):
            if i % 10 == 0:
                logger.info(f"Calculating reward for episode {i+1}/{len(episodes)}")
            
            reward = self.calculate_reward(episode)
            rewards.append(reward)
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        logger.info(f"Batch reward calculation completed. Average reward: {avg_reward:.3f}")
        
        return rewards
    
    def create_wrong_decision_prompt(self, task_context: Dict[str, Any]) -> str:
        """Create a prompt specifically designed to trigger wrong decision scenarios"""
        
        # Common wrong decision patterns based on error analysis
        wrong_decision_scenarios = [
            "Ask about order status but don't specify which order when multiple exist",
            "Request cancellation but be unclear about which items to cancel",
            "Ask for product information but don't provide enough details",
            "Request modification of pending order without specifying which items",
            "Ask about return policy but don't mention specific purchase",
            "Request tracking information without providing order details",
            "Ask to change delivery address but be vague about which order",
            "Request price adjustment without specifying which product"
        ]
        
        import random
        scenario = random.choice(wrong_decision_scenarios)
        
        prompt = f"""
You are a customer service user simulator. Your goal is to create a challenging scenario that might cause the agent to make wrong decisions about which tools to call.

Scenario: {scenario}

Context: {json.dumps(task_context, indent=2)}

Create a user interaction that starts clear but becomes ambiguous in a way that might lead the agent to:
1. Call the wrong type of tool for the situation
2. Make assumptions instead of asking clarifying questions  
3. Miss domain-specific rules or requirements
4. Choose inefficient tool sequences

The scenario should be realistic and test the agent's decision-making capabilities.
"""
        
        return prompt
