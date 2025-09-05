# RL Training for Gemma3 with PPO and Gemini-2.5-pro Reward Model

This directory contains a complete implementation for training Gemma3 using Proximal Policy Optimization (PPO) with Gemini-2.5-pro as the reward model, specifically designed to address "wrong decision" errors identified in tau-bench evaluations.

## 🎯 Overview

Based on error analysis showing that **81.5% of failures** are due to "wrong decision" errors, this training setup:

- Uses **PPO (Proximal Policy Optimization)** to train Gemma3-12B
- Employs **Gemini-2.5-pro as reward model** for intelligent evaluation
- Focuses on **scenarios that trigger wrong decisions**
- Utilizes **all 8 H100 GPUs** with tensor parallelism
- Provides **seamless integration** back into tau-bench for evaluation

## 📁 Project Structure

```
rl_training/
├── ppo_trainer.py              # Main PPO training implementation
├── reward_model.py             # Gemini-2.5-pro reward model
├── tau_bench_data_collector.py # Data collection from tau-bench
├── ppo_utils.py                # PPO utilities and configurations
├── integrate_model.py          # Model integration script
├── setup_rl_training.sh        # Environment setup script
├── requirements.txt            # Python dependencies
├── configs/                    # Configuration files
│   ├── ppo_config.yaml        # PPO training configuration
│   └── accelerate_config.yaml # Multi-GPU acceleration config
├── outputs/                    # Training outputs and checkpoints
├── logs/                       # Training logs
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Run the setup script (this will install all dependencies)
./rl_training/setup_rl_training.sh

# Set up Google Cloud credentials
export GOOGLE_CLOUD_PROJECT=your-project-id
gcloud auth application-default login
```

### 2. Start Training

```bash
# Launch PPO training with 8 H100 GPUs
./rl_training/launch_training.sh
```

### 3. Monitor Training

```bash
# In another terminal, monitor GPU usage and training metrics
python rl_training/monitor_training.py
```

### 4. Deploy and Evaluate Trained Model

```bash
# After training completes, integrate the model
python rl_training/integrate_model.py rl_training/outputs/final_model

# Run complete evaluation pipeline
./rl_training/run_complete_evaluation.sh
```

## 🎛️ Configuration

### Training Configuration (`configs/ppo_config.yaml`)

```yaml
model:
  name: "google/gemma-2-9b-it"  # Base model
  use_flash_attention: true
  
training:
  num_episodes: 50              # Number of training episodes
  rollouts_per_episode: 100     # Rollouts per episode
  learning_rate: 1.0e-5         # Learning rate
  ppo_epochs: 4                 # PPO optimization epochs
  clip_epsilon: 0.2             # PPO clipping parameter
  
hardware:
  gpu_count: 8                  # Number of GPUs
  tensor_parallel_size: 8       # Tensor parallelism
  gpu_memory_utilization: 0.9   # GPU memory usage
```

### Hardware Requirements

- **8 H100 GPUs** (recommended)
- **512GB+ RAM**
- **2TB+ Storage** for model checkpoints
- **CUDA 12.1+**
- **Python 3.9+**

## 🧠 Technical Details

### PPO Training Process

1. **Data Collection**: Generate rollouts using current policy on tau-bench tasks
2. **Reward Calculation**: Use Gemini-2.5-pro to evaluate each trajectory
3. **Policy Update**: Update policy using PPO with computed advantages
4. **Value Function**: Train critic to estimate state values
5. **Iteration**: Repeat for specified number of episodes

### Policy Rollout Data Collection

**Rollout Data Generation Location:**
- **File**: `rl_training/tau_bench_data_collector.py` (Lines 207-280)
- **Method**: `generate_episode()` 
- **Content**: Mock conversation trajectories for PPO training data
- **Usage**: Called during PPO training to collect rollout experiences

**Current Episode Data Structure:**

```python
episode = {
    "trajectory": [
        {
            "role": "user",
            "content": "Hi, I need help with my order."
        },
        {
            "role": "assistant", 
            "content": "I'd be happy to help you with your order. Can you please provide your order number?"
        },
        {
            "role": "user",
            "content": "It's order #12345. I want to cancel it."
        },
        {
            "role": "assistant",
            "content": "I can help you cancel order #12345. Let me process that for you right away."
        }
    ],
    "reward": 0.75,  # Calculated by Gemini-2.5-pro reward model
    "task_id": "mock_task_42",
    "success": True
}
```

⚠️ **Note**: Currently using mock episodes for testing. The full tau-bench integration with actual task scenarios and tool calling is planned for future implementation. 

For detailed information about tau-bench conversation flow and tool calling mechanisms, see [`../TAU_BENCH_CONVERSATION_FLOW.md`](../TAU_BENCH_CONVERSATION_FLOW.md).

### Reward Model Design

The Gemini-2.5-pro reward model evaluates:

- **Decision Making Quality (40%)**: Correct tool selection and sequence
- **Information Accuracy (30%)**: Proper use of tool arguments
- **Task Completion (20%)**: Successful resolution of user request
- **User Experience (10%)**: Communication quality

**Reward Model Prompt Location:**
- **File**: `rl_training/reward_model.py` (Lines 50-140)
- **Method**: `create_reward_prompt()` 
- **Content**: Detailed evaluation criteria with weighted scoring and one-shot examples
- **Usage**: Called during PPO training to evaluate each trajectory with Gemini-2.5-pro

**Complete Reward Model Prompt:**

**Reward Model Prompt Location:**
- **File**: `rl_training/reward_model.py` (Lines 50-140)
- **Method**: `create_reward_prompt()` 
- **Content**: Detailed evaluation criteria with weighted scoring and one-shot examples
- **Usage**: Called during PPO training to evaluate each trajectory with Gemini-2.5-pro

⚠️ **Note**: The examples below contain synthetic data that may not align with actual tau-bench ground truth. The reward model example uses `get_customer_info` but tau-bench actually uses `find_user_id_by_name_zip` and `get_user_details`. This should be validated against real tau-bench task data.

```
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
2. ASSISTANT (tool calls): [{"name": "get_customer_info", "arguments": {"customer_name": "Sofia Martinez", "zip_code": "90210"}}]
3. SYSTEM: Returns customer info with order IDs [ABC123, DEF456]
4. ASSISTANT (tool calls): [{"name": "get_order_details", "arguments": {"order_id": "ABC123"}}]
5. SYSTEM: Returns order details showing grill + other items
6. ASSISTANT (tool calls): [{"name": "cancel_order", "arguments": {"order_id": "ABC123", "reason": "Customer request"}}]
7. ASSISTANT: "Perfect! I've cancelled your grill order ABC123. Your other orders remain active."

Why this gets 1.0: Perfect decision making (correct tool sequence), accurate use of provided information, complete task resolution, excellent user communication.

**NOW EVALUATE THIS TASK:**

**User Instruction:** {gt_instruction}

**Expected Ground Truth Actions:** {gt_actions}

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
```

### Wrong Decision Focus

Training specifically targets scenarios that commonly lead to wrong decisions:

- Ambiguous cancellation requests
- Missing product lookups
- Wrong customer identification methods
- Incomplete order modifications
- Flight booking confusion
- Baggage policy misunderstandings

### LoRA Fine-tuning

Uses **LoRA (Low-Rank Adaptation)** for efficient training:
- **Rank**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target modules**: All attention and MLP layers

## 📊 Training Monitoring

### Weights & Biases Integration

Training metrics are automatically logged to W&B:
- Policy loss
- Value loss
- KL divergence
- Average reward
- Success rate

### Resource Monitoring

Use the monitoring script to track:
- GPU utilization and memory usage
- CPU and RAM consumption
- Training progress and metrics
- Temperature monitoring

### Key Metrics to Watch

- **Average Reward**: Should increase over time
- **KL Divergence**: Should stay below 0.1
- **Success Rate**: Improvement in task completion
- **Policy Loss**: Should decrease and stabilize

## 🔧 Model Integration

After training, integrate the model back into tau-bench:

```bash
# Integrate trained model
python rl_training/integrate_model.py rl_training/outputs/final_model --name my-trained-gemma3

# Deploy with vLLM
./rl_training/deploy_trained_model.sh

# Use in tau-bench
python run.py \
    --model my-trained-gemma3 \
    --model-provider vllm \
    --env retail \
    --agent-strategy tool-calling
```

## 📈 Expected Results

Based on error analysis, training should improve:

1. **Wrong Decision Errors**: Reduction from 81.5% to <50%
2. **Overall Success Rate**: Improvement from 6.1% to 15-25%
3. **Tool Selection**: Better understanding of domain rules
4. **Argument Usage**: More accurate parameter filling

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `gpu_memory_utilization` in config
   - Decrease `rollouts_per_episode`
   - Use gradient checkpointing

2. **Model Server Won't Start**
   - Check port availability (8001)
   - Verify model path exists
   - Check CUDA drivers

3. **Slow Training**
   - Verify all 8 GPUs are utilized
   - Check tensor parallelism settings
   - Monitor GPU utilization

4. **Reward Model Errors**
   - Verify Google Cloud credentials
   - Check `GOOGLE_CLOUD_PROJECT` environment variable
   - Ensure Vertex AI API is enabled

### Performance Optimization

- Use **Flash Attention 2** for faster training
- Enable **gradient checkpointing** for memory efficiency
- Use **mixed precision** (bf16) for speed
- Optimize **batch sizes** based on available memory

## 📚 Advanced Usage

### Custom Scenarios

Add custom wrong decision scenarios in `tau_bench_data_collector.py`:

```python
def create_custom_scenario(self):
    return {
        "instruction": "Your custom instruction",
        "expected_actions": [...],
        "wrong_decision_triggers": [...]
    }
```

### Reward Function Tuning

Modify reward weights in `reward_model.py`:

```python
# Adjust evaluation criteria weights
DECISION_MAKING_WEIGHT = 0.4  # Default: 40%
INFORMATION_ACCURACY_WEIGHT = 0.3  # Default: 30%
TASK_COMPLETION_WEIGHT = 0.2  # Default: 20%
USER_EXPERIENCE_WEIGHT = 0.1  # Default: 10%
```

## 🔬 Experimental Features

### Curriculum Learning

Gradually increase task difficulty:
- Start with simple scenarios
- Progress to complex multi-step tasks
- Introduce edge cases

### Reward Shaping

Additional reward signals:
- Intermediate step rewards
- Efficiency bonuses
- Exploration incentives

### Model Distillation

Use trained model to teach smaller models:
- Gemma-7B student model
- Knowledge transfer techniques
- Deployment efficiency


## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Tools and Libraries Used

- **Google** for Gemma and Gemini models
- **Hugging Face** for transformers library
- **vLLM** for efficient model serving
- **Weights & Biases** for experiment tracking
- **tau-bench** team for the evaluation framework
