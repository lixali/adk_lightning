# Tau-Bench Conversation Information Flow

This document explains how conversation information flows through the tau-bench system and how message histories are constructed and fed to both assistant and user simulation models during evaluation.

## Overview

Tau-bench maintains complete conversation histories for both the assistant agent and user simulation models. Each model receives the full dialogue context when making decisions, ensuring they have complete situational awareness throughout multi-turn conversations.

## Core Components

### 1. ToolCallingAgent - Assistant Model Message Management

The `ToolCallingAgent` class maintains the assistant's conversation history in a `messages` list that grows with each interaction.

```python
# From tau_bench/agents/tool_calling_agent.py lines 54-87

def solve(self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
    # Initialize messages with system prompt and initial user message
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": self.wiki},
        {"role": "user", "content": obs},
    ]
    
    for _ in range(max_num_steps):
        # Pass COMPLETE conversation history to model
        res = completion(
            messages=messages,  # Full conversation context
            model=self.litellm_model,
            custom_llm_provider=self.litellm_provider,
            tools=self.tools_info,
            temperature=self.temperature,
        )
        
        next_message = res.choices[0].message.model_dump()
        action = message_to_action(next_message)
        env_response = env.step(action)
        
        # Add assistant response and tool results to conversation history
        if action.name != RESPOND_ACTION_NAME:
            # Tool call: Add both the tool call and tool result
            next_message["tool_calls"] = next_message["tool_calls"][:1]
            messages.extend([
                next_message,  # Assistant's tool call
                {
                    "role": "tool",
                    "tool_call_id": next_message["tool_calls"][0]["id"],
                    "name": next_message["tool_calls"][0]["function"]["name"],
                    "content": env_response.observation,  # Tool execution result
                },
            ])
        else:
            # Regular response: Add assistant message and user reply
            messages.extend([
                next_message,  # Assistant's response
                {"role": "user", "content": env_response.observation},  # User's reply
            ])
```

**Key Points:**
- The `messages` list starts with system prompt and initial user message
- Every `completion()` call receives the **complete conversation history**
- After each interaction, new messages are appended to maintain full context
- Tool calls and their results are both preserved in the conversation

### 2. User Simulation Models - Maintaining Separate Conversation History

User simulation models also maintain their own complete conversation histories and receive full context when generating responses.

```python
# From tau_bench/envs/user.py lines 50-80 (LLMUserSimulationEnv)

class LLMUserSimulationEnv(UserSimulationEnv):
    def __init__(self, model: str, provider: str, temperature: float = 0.0):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.messages: List[Dict[str, Any]] = []  # User's conversation history
        
    def reset(self, task_index: Optional[int] = None) -> EnvResetResult:
        # Initialize user's conversation history
        self.messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": self.get_initial_message()}
        ]
        
    def step(self, action: Action) -> EnvStepResult:
        # Add assistant's action to user's conversation history
        if action.name == RESPOND_ACTION_NAME:
            self.messages.append({
                "role": "assistant", 
                "content": action.kwargs["response"]
            })
        else:
            # Add tool call information to user's context
            self.messages.append({
                "role": "assistant",
                "content": f"I'm using {action.name} tool with parameters: {action.kwargs}"
            })
            
        # Generate user response with COMPLETE conversation history
        completion_result = completion(
            messages=self.messages,  # Full user conversation context
            model=self.model,
            custom_llm_provider=self.provider,
            temperature=self.temperature,
        )
        
        user_response = completion_result.choices[0].message.content
        
        # Add user's own response to their conversation history
        self.messages.append({
            "role": "user", 
            "content": user_response
        })
        
        return EnvStepResult(
            observation=user_response,
            reward=self.calculate_reward(),
            done=self.is_done(),
            info={}
        )
```

**Key Points:**
- User simulation maintains separate `self.messages` conversation history
- User model receives **complete conversation context** including all assistant actions
- User's own responses are added back to their conversation history
- Both tool calls and regular responses are tracked in user's context

### 3. Environment Coordination - Base Environment

The base environment coordinates message flow between assistant and user simulation models.

```python
# From tau_bench/envs/base.py lines 70-100

class Env:
    def step(self, action: Action) -> EnvStepResult:
        if action.name == RESPOND_ACTION_NAME:
            # Assistant is responding to user - pass to user simulation
            user_response = self.user_sim_env.step(action)
            observation = user_response.observation
        else:
            # Assistant is calling a tool - execute and return result
            observation = self.execute_tool(action)
            
        # Check if conversation should end
        done = "###STOP###" in observation if observation else False
        
        # Calculate reward and update info
        reward = self.calculate_reward() if done else 0.0
        info = self.get_current_info()
        
        return EnvStepResult(
            observation=observation,
            reward=reward, 
            done=done,
            info=info
        )
```

## Message Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Assistant      │    │   Environment    │    │  User Simulation│
│  Agent          │    │   Coordinator    │    │  Model          │
│                 │    │                  │    │                 │
│ messages: [     │    │                  │    │ messages: [     │
│   {system},     │    │                  │    │   {system},     │
│   {user},       │◄───┤                  │    │   {user},       │
│   {assistant},  │    │                  │    │   {assistant},  │
│   {tool},       │    │                  │    │   {user},       │
│   {user},       │    │                  │    │   ...           │
│   ...           │    │                  │    │ ]               │
│ ]               │    │                  │    │                 │
│                 │    │                  │    │                 │
│ completion() ───┼────┤ action ──────────┼────┤ completion()    │
│                 │    │                  │    │                 │
│                 │    │ tool_result      │    │                 │
│                 │◄───┤     OR           │◄───┤ user_response   │
│                 │    │ user_response    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Conversation History Structure

Both models maintain conversation histories as lists of dictionaries in OpenAI chat format:

### Assistant Model Message Structure
```python
messages = [
    {"role": "system", "content": "# Retail agent policy\n..."},
    {"role": "user", "content": "Hello, I need help with an order."},
    {"role": "assistant", "content": "I'd be happy to help...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_123", "name": "get_order", "content": "Order details..."},
    {"role": "user", "content": "Great! Now I need to..."},
    {"role": "assistant", "content": "I can help with that..."},
    # ... conversation continues
]
```

### User Simulation Model Message Structure
```python
messages = [
    {"role": "system", "content": "You are Yusuf Rossi in 19122..."},
    {"role": "user", "content": "Hello, I need help with an order."},
    {"role": "assistant", "content": "I'd be happy to help..."},
    {"role": "user", "content": "Great! Now I need to..."},
    {"role": "assistant", "content": "I can help with that..."},
    # ... conversation continues from user's perspective
]
```

## Key Implementation Details

### 1. Complete Context Preservation
- **Both models receive complete conversation history** on every turn
- No context truncation or summarization occurs
- All tool calls, tool results, and responses are preserved
- System prompts remain at the beginning of each conversation

### 2. Message Synchronization
- Assistant model tracks tool executions and their results
- User model tracks assistant responses and actions
- Environment coordinates message passing between models
- Both models stay synchronized on conversation state

### 3. Tool Call Handling
- Assistant model records both tool calls and their results
- User model is informed about tool usage through formatted messages
- Tool results are immediately available to assistant for next response
- User model can respond to tool execution outcomes

### 4. Conversation Termination
- Conversations end when user simulation returns "###STOP###"
- Both models have complete context when final evaluation occurs
- Final states are compared using SHA256 hashing for reward calculation

## Example: Failed Gemma3 Conversation Analysis

Looking at the failed Gemma3 conversation in the attached file, we can see the conversation flow:

```python
# Task 0 trajectory shows the problem:
[
    {"role": "system", "content": "# Retail agent policy..."},
    {"role": "user", "content": "Hello there! I recently received an order..."},
    
    # Gemma3 fails to use proper tool calling format:
    {"content": "{\"tool_name\": \"get_product_details\", \"parameters\": {\"product_id\": \"6086499569\"}}", 
     "role": "assistant", "tool_calls": null, "function_call": null},
    
    # User model receives complete context and responds appropriately:
    {"role": "user", "content": "Alright, I can certainly help with exchanges..."},
    
    # Pattern continues with Gemma3 using wrong format while user model works correctly
    # ...
]
```

**The key insight**: Both models receive complete conversation context, but Gemma3 fails due to improper tool calling format, not due to missing conversation context.

## Conclusion

Tau-bench's conversation flow ensures that:

1. **Complete Context**: Both assistant and user models receive full conversation history
2. **Proper Synchronization**: Message flow is coordinated through the environment
3. **Format Compliance**: Standard OpenAI chat format is used throughout
4. **State Preservation**: All interactions are recorded and maintained
5. **Evaluation Accuracy**: Final states reflect complete conversation outcomes

The system's design ensures that model failures (like those seen with Gemma3) are due to model limitations rather than insufficient context or conversation flow issues.
