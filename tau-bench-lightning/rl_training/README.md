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

### Reward Model Design

The Gemini-2.5-pro reward model evaluates:

- **Decision Making Quality (40%)**: Correct tool selection and sequence
- **Information Accuracy (30%)**: Proper use of tool arguments
- **Task Completion (20%)**: Successful resolution of user request
- **User Experience (10%)**: Communication quality

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

### Multi-Environment Training

Train on both retail and airline environments:

```yaml
data:
  environments: ["retail", "airline"]
  environment_weights: [0.7, 0.3]  # 70% retail, 30% airline
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

## 📄 Citation

If you use this training setup in your research, please cite:

```bibtex
@software{gemma3_ppo_training,
  title={PPO Training for Gemma3 with Gemini-2.5-pro Reward Model},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/tau-bench-lightning}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google** for Gemma and Gemini models
- **Hugging Face** for transformers library
- **vLLM** for efficient model serving
- **Weights & Biases** for experiment tracking
- **tau-bench** team for the evaluation framework
