#!/usr/bin/env python3
"""
PPO Training for Gemma3 with Gemini-2.5-pro Reward Model
This script implements PPO training using all 8 H100 GPUs.
"""

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import wandb
import signal
import sys

# Import our reward model and environment
from reward_model import GeminiRewardModel
from tau_bench_data_collector import TauBenchDataCollector
from ppo_utils import PPOConfig, PPOMemory, compute_advantages

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GracefulShutdownHandler:
    """Handle graceful shutdown for distributed training"""
    def __init__(self):
        self.shutdown_requested = False
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, requesting graceful shutdown...")
        self.shutdown_requested = True
    
    def should_shutdown(self):
        """Check if shutdown was requested"""
        return self.shutdown_requested

class PPOTrainer:
    def __init__(self, config: PPOConfig):
        self.config = config
        # Initialize accelerator with DDP configuration for unused parameters
        self.accelerator = Accelerator(
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )
        
        # Initialize graceful shutdown handler
        self.shutdown_handler = GracefulShutdownHandler()
        
        # Initialize tokenizer and models
        self.setup_models()
        
        # Initialize reward model (Gemini-2.5-pro)
        self.reward_model = GeminiRewardModel()
        
        # Initialize data collector
        self.data_collector = TauBenchDataCollector(
            model_name=config.model_name,
            tokenizer=self.tokenizer
        )
        
        # PPO memory buffer
        self.memory = PPOMemory()
        
        # Training metrics
        self.metrics = {
            'epoch': 0,
            'total_episodes': 0,
            'avg_reward': 0.0,
            'avg_policy_loss': 0.0,
            'avg_value_loss': 0.0,
            'kl_divergence': 0.0
        }
        
        # Initialize wandb for logging
        if self.accelerator.is_main_process and self.config.use_wandb:
            wandb.init(
                project="gemma3-tau-bench-ppo",
                config=config.__dict__,
                name=f"ppo_gemma3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        elif self.accelerator.is_main_process:
            logger.info("Wandb logging disabled for test run")
    
    def setup_models(self):
        """Setup Gemma3 model with LoRA for efficient training"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model_kwargs = {
            "dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        
        # Don't use device_map with accelerate distributed training
        if not self.accelerator.state.distributed_type.value == "MULTI_GPU":
            model_kwargs["device_map"] = "auto"
        
        # Only add flash attention if enabled and available
        if self.config.use_flash_attention:
            try:
                # Test if flash attention is available
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash attention 2")
            except ImportError as e:
                logger.warning(f"Flash attention not available, using default attention: {e}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Setup LoRA configuration for efficient training
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # For memory efficiency, we'll use the current model as reference 
        # instead of loading a separate reference model
        
        # For memory efficiency, use the same model for critic but with separate value head
        # This avoids loading a second copy of the 12B model
        self.critic = self.model
        
        # Add value head to critic
        self.critic.config.num_labels = 1
        # Handle different config structures (Gemma3 has text_config subconfig)
        hidden_size = getattr(self.critic.config, 'hidden_size', None)
        if hidden_size is None and hasattr(self.critic.config, 'text_config'):
            hidden_size = self.critic.config.text_config.hidden_size
        if hidden_size is None:
            raise ValueError(f"Could not determine hidden_size for model {self.config.model_name}")
            
        # Create value head  
        self.value_head = torch.nn.Linear(hidden_size, 1)
        
        # Move value head to same device and dtype as critic
        device = next(self.critic.parameters()).device
        dtype = next(self.critic.parameters()).dtype
        self.value_head = self.value_head.to(device=device, dtype=dtype)
        
        # Setup optimizers
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Include both critic and value head parameters
        critic_params = list(self.critic.parameters()) + list(self.value_head.parameters())
        self.critic_optimizer = torch.optim.AdamW(
            critic_params,
            lr=self.config.critic_learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Prepare models with accelerator (since critic = model, only prepare once)
        self.model, self.value_head, self.optimizer, self.critic_optimizer = self.accelerator.prepare(
            self.model, self.value_head, self.optimizer, self.critic_optimizer
        )
        # Update critic reference to prepared model
        self.critic = self.model
    
    def collect_rollouts(self, num_rollouts: int) -> List[Dict[str, Any]]:
        """Collect rollouts using the current policy"""
        logger.info(f"Collecting {num_rollouts} rollouts...")
        
        rollouts = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(num_rollouts):
                # Check for graceful shutdown during rollout collection
                if self.shutdown_handler.should_shutdown():
                    logger.info(f"Graceful shutdown requested during rollout collection. Collected {i} rollouts so far.")
                    break
                    
                if i % 10 == 0:
                    logger.info(f"Collecting rollout {i+1}/{num_rollouts}")
                
                # Generate episode using current policy
                episode_data = self.data_collector.generate_episode(
                    self.model,
                    focus_on_wrong_decisions=True  # Focus on scenarios that trigger wrong decisions
                )
                
                # Calculate reward using Gemini-2.5-pro
                reward = self.reward_model.calculate_reward(episode_data)
                
                episode_data['reward'] = reward
                rollouts.append(episode_data)
        
        logger.info(f"Collected {len(rollouts)} rollouts with avg reward: {np.mean([r['reward'] for r in rollouts]):.3f}")
        return rollouts
    
    def compute_policy_loss(self, states, actions, advantages, old_log_probs):
        """Compute PPO policy loss with clipping"""
        # Get current policy outputs (without using labels to avoid shape mismatch)
        outputs = self.model(states)
        
        # Get log probabilities for the last token (where we make decisions)
        logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        current_log_probs = F.log_softmax(logits, dim=-1)
        
        # For simplicity, compute log prob of the first token in actions
        # In a full implementation, you'd want to properly align actions with states
        action_tokens = actions[:, 0]  # Take first token of each action sequence
        current_action_log_probs = current_log_probs.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Compute probability ratio (use old_log_probs as reference)
        ratio = torch.exp(current_action_log_probs - old_log_probs)
        
        # Compute clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Simple KL divergence penalty (current vs old log probs)
        kl_div = (current_action_log_probs - old_log_probs).mean()
        
        total_loss = policy_loss + self.config.kl_coeff * kl_div
        
        return total_loss, policy_loss.item(), kl_div.item()
    
    def compute_value_loss(self, states, returns):
        """Compute value function loss"""
        # Get critic outputs
        value_outputs = self.critic(states, output_hidden_states=True)
        
        # Get the last hidden state from the text model
        if hasattr(value_outputs, 'hidden_states') and value_outputs.hidden_states is not None:
            # Use the last layer's hidden state
            last_hidden = value_outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_size]
        else:
            # Fallback: use mean of logits
            last_hidden = value_outputs.logits[:, -1, :].mean(dim=-1, keepdim=True)
        
        # Pass through value head to get scalar values
        values = self.value_head(last_hidden).squeeze(-1)
        
        # Ensure same dtype for loss computation
        returns = returns.to(values.dtype)
        
        # Compute MSE loss
        value_loss = F.mse_loss(values, returns)
        
        return value_loss, values.detach()
    
    def train_step(self, rollouts: List[Dict[str, Any]]):
        """Perform one PPO training step with gradient accumulation"""
        # Process rollouts into training data
        states, actions, rewards, log_probs = [], [], [], []
        
        for rollout in rollouts:
            # Convert trajectory to training format
            trajectory_data = self.data_collector.process_rollout_for_training(rollout)
            states.extend(trajectory_data['states'])
            actions.extend(trajectory_data['actions'])
            rewards.extend(trajectory_data['rewards'])
            log_probs.extend(trajectory_data['log_probs'])
        
        # Check if we have valid data
        if not states:
            logger.warning("No valid training data collected, skipping training step")
            return
        
        # Use pad_sequence for robust padding
        from torch.nn.utils.rnn import pad_sequence
        
        # Pad sequences to the same length (removed max_sequence_length limit for better performance)
        padded_states = pad_sequence(states, batch_first=True, padding_value=0)
        padded_actions = pad_sequence(actions, batch_first=True, padding_value=0)
        
        # Convert to correct device and dtype
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        states = padded_states.to(device)
        actions = padded_actions.to(device)
        rewards = torch.tensor(rewards, dtype=dtype).to(device)
        
        # Compute initial log probabilities for the collected states and actions
        with torch.no_grad():
            initial_outputs = self.model(states)
            initial_logits = initial_outputs.logits[:, -1, :]
            initial_log_probs = F.log_softmax(initial_logits, dim=-1)
            action_tokens = actions[:, 0]
            old_log_probs = initial_log_probs.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Compute advantages and returns
        with torch.no_grad():
            _, values = self.compute_value_loss(states, rewards)
            advantages, returns = compute_advantages(
                rewards, values, self.config.gamma, self.config.gae_lambda
            )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get gradient accumulation steps
        grad_accum_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        # Calculate effective batch size for accumulation
        batch_size = states.size(0)
        accumulation_batch_size = max(1, batch_size // grad_accum_steps)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_kl_div = 0
        
        for epoch in range(self.config.ppo_epochs):
            # Clear gradients at the start of each epoch
            self.optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_kl_div = 0
            
            # Process data in smaller chunks for gradient accumulation
            for i in range(0, batch_size, accumulation_batch_size):
                end_idx = min(i + accumulation_batch_size, batch_size)
                
                # Get batch slice
                batch_states = states[i:end_idx]
                batch_actions = actions[i:end_idx]
                batch_advantages = advantages[i:end_idx]
                batch_old_log_probs = old_log_probs[i:end_idx]
                batch_returns = returns[i:end_idx]
                
                # Skip if batch is empty
                if batch_states.size(0) == 0:
                    continue
                
                # Forward pass for both policy and value with the same model
                # This ensures all model parameters participate in the computation
                with self.accelerator.accumulate(self.model):
                    # Single forward pass for both policy and value computations
                    # to ensure all parameters receive gradients
                    
                    # Policy computation 
                    outputs = self.model(batch_states, output_hidden_states=True)
                    logits = outputs.logits[:, -1, :]
                    action_log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get action log probabilities
                    action_tokens = batch_actions[:, 0] if batch_actions.size(1) > 0 else torch.zeros(batch_actions.size(0), dtype=torch.long, device=batch_actions.device)
                    current_action_log_probs = action_log_probs.gather(-1, action_tokens.unsqueeze(-1)).squeeze(-1)
                    
                    # Compute policy loss components
                    ratio = torch.exp(current_action_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    kl_div = (current_action_log_probs - batch_old_log_probs).mean()
                    
                    # Value computation using the same forward pass outputs
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        # Use the last layer's hidden state, last token
                        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_size]
                    else:
                        # Fallback: need to get proper hidden representation
                        # Use the model's embeddings to get hidden size
                        batch_size = batch_states.size(0)
                        hidden_size = self.model.config.hidden_size if hasattr(self.model.config, 'hidden_size') else self.model.config.text_config.hidden_size
                        last_hidden = outputs.logits[:, -1, :hidden_size]  # Take first hidden_size dims from logits
                    
                    # Ensure proper shape for value head input
                    if last_hidden.dim() == 1:
                        last_hidden = last_hidden.unsqueeze(0)  # Add batch dimension if missing
                    
                    # Pass through value head to get scalar values
                    values = self.value_head(last_hidden).squeeze(-1)
                    
                    # Ensure same dtype for loss computation
                    batch_returns = batch_returns.to(values.dtype)
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # Combine losses to ensure all parameters receive gradients
                    total_loss = policy_loss + value_loss + self.config.kl_coeff * kl_div
                    
                    # Scale loss by accumulation steps
                    scaled_total_loss = total_loss / grad_accum_steps
                    self.accelerator.backward(scaled_total_loss)
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_kl_div += kl_div.item()
            
            # Apply gradients after accumulation
            if self.accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.critic_optimizer.step()
            
            # Average losses across accumulation steps
            num_batches = (batch_size + accumulation_batch_size - 1) // accumulation_batch_size
            total_policy_loss += epoch_policy_loss / num_batches
            total_value_loss += epoch_value_loss / num_batches
            total_kl_div += epoch_kl_div / num_batches
        
        # Update metrics
        self.metrics['avg_policy_loss'] = total_policy_loss / self.config.ppo_epochs
        self.metrics['avg_value_loss'] = total_value_loss / self.config.ppo_epochs
        self.metrics['kl_divergence'] = total_kl_div / self.config.ppo_epochs
        self.metrics['avg_reward'] = np.mean([r['reward'] for r in rollouts])
    
    def train(self):
        """Main training loop"""
        logger.info("Starting PPO training...")
        
        for episode in range(self.config.num_episodes):
            # Check for graceful shutdown
            if self.shutdown_handler.should_shutdown():
                logger.info("Graceful shutdown requested. Saving checkpoint and exiting...")
                self.save_checkpoint(episode)
                break
                
            logger.info(f"\n=== Episode {episode + 1}/{self.config.num_episodes} ===")
            
            # Collect rollouts
            rollouts = self.collect_rollouts(self.config.rollouts_per_episode)
            
            # Check for shutdown again after rollout collection
            if self.shutdown_handler.should_shutdown():
                logger.info("Graceful shutdown requested after rollout collection. Saving checkpoint and exiting...")
                self.save_checkpoint(episode)
                break
            
            # Train on collected data
            self.train_step(rollouts)
            
            # Update metrics
            self.metrics['epoch'] = episode + 1
            self.metrics['total_episodes'] = episode + 1
            
            # Log metrics
            if self.accelerator.is_main_process:
                logger.info(f"Avg Reward: {self.metrics['avg_reward']:.3f}")
                logger.info(f"Policy Loss: {self.metrics['avg_policy_loss']:.6f}")
                logger.info(f"Value Loss: {self.metrics['avg_value_loss']:.6f}")
                logger.info(f"KL Divergence: {self.metrics['kl_divergence']:.6f}")
                
            if self.config.use_wandb and self.accelerator.is_main_process:
                wandb.log(self.metrics)
            
            # Save checkpoint
            if (episode + 1) % self.config.save_every == 0:
                self.save_checkpoint(episode + 1)
            
            # Early stopping based on KL divergence
            if self.metrics['kl_divergence'] > self.config.max_kl:
                logger.warning(f"KL divergence {self.metrics['kl_divergence']:.6f} exceeded max {self.config.max_kl}. Stopping training.")
                break
        
        # Final cleanup and save
        if self.accelerator.is_main_process:
            if self.shutdown_handler.should_shutdown():
                logger.info("Training completed due to graceful shutdown request.")
            else:
                logger.info("Training completed successfully!")
        
        # Save final model in both LoRA and merged formats
        # save_merged=True creates both LoRA adapters (274MB) and full merged model (~24GB)
        self.save_final_model(save_merged=True)
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint - LoRA adapters only to save space"""
        checkpoint_dir = f"{self.config.output_dir}/checkpoint-{episode}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save only LoRA adapters (not the full 23GB model)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(checkpoint_dir)
        
        # Save optimizer states and other training info
        # Note: This still saves optimizer states which can be large
        optimizer_state = {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode': episode,
            'metrics': self.metrics
        }
        
        torch.save(optimizer_state, f"{checkpoint_dir}/optimizer_state.pt")
        
        # Save metrics separately for easy access
        with open(f"{checkpoint_dir}/metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"LoRA checkpoint saved to {checkpoint_dir} (much smaller than full model)")
        
        # Optional: Clean up old full-model checkpoints to save space
        self._cleanup_old_checkpoints(episode)
    
    def _cleanup_old_checkpoints(self, current_episode: int, keep_last: int = 2):
        """Clean up old checkpoints to save disk space"""
        import glob
        checkpoint_pattern = f"{self.config.output_dir}/checkpoint-*"
        checkpoints = glob.glob(checkpoint_pattern)
        
        # Sort by episode number
        def extract_episode(path):
            try:
                return int(path.split('-')[-1])
            except:
                return 0
        
        checkpoints.sort(key=extract_episode)
        
        # Keep only the last N checkpoints
        if len(checkpoints) > keep_last:
            to_remove = checkpoints[:-keep_last]
            for checkpoint_dir in to_remove:
                # Only remove if it contains the large model.safetensors
                large_model_file = os.path.join(checkpoint_dir, "model.safetensors")
                if os.path.exists(large_model_file) and os.path.getsize(large_model_file) > 1e9:  # > 1GB
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Removed old large checkpoint: {checkpoint_dir}")
    
    def save_final_model(self, save_merged=True):
        """Save the final trained model in both LoRA and merged formats"""
        final_dir = f"{self.config.output_dir}/final_model"
        os.makedirs(final_dir, exist_ok=True)
        
        # Save LoRA adapters (efficient format)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"LoRA adapters saved to {final_dir}")
        
        # Optionally save merged full model
        if save_merged:
            merged_dir = f"{self.config.output_dir}/merged_model"
            os.makedirs(merged_dir, exist_ok=True)
            
            logger.info("Creating merged model (this may take a few minutes)...")
            
            # Merge LoRA adapters with base model
            try:
                merged_model = unwrapped_model.merge_and_unload()
                merged_model.save_pretrained(
                    merged_dir,
                    save_method="safetensors",
                    max_shard_size="5GB"
                )
                self.tokenizer.save_pretrained(merged_dir)
                logger.info(f"Merged model saved to {merged_dir}")
                
                # Create Ollama Modelfile
                self.create_ollama_modelfile(merged_dir)
                
            except Exception as e:
                logger.warning(f"Could not create merged model: {e}")
                logger.info("LoRA adapters are still available for inference")
        
        # Save configuration
        config_dict = {
            **self.config.__dict__,
            'final_metrics': self.metrics,
            'training_completed': datetime.now().isoformat(),
            'model_formats': {
                'lora_adapters': final_dir,
                'merged_model': f"{self.config.output_dir}/merged_model" if save_merged else None
            }
        }
        
        with open(f"{final_dir}/training_config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Final model saved to {final_dir}")
        
        # Create deployment script
        self.create_deployment_script(final_dir)
    
    def create_ollama_modelfile(self, model_dir: str):
        """Create Ollama Modelfile for the merged model"""
        modelfile_content = f"""FROM {model_dir}

# Set custom parameters optimized for your trained model
PARAMETER temperature {self.config.temperature}
PARAMETER top_p 0.9
PARAMETER top_k 40

# Set custom template (adjust based on your training format)
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
{{{{ end }}}}{{{{ .Response }}}}<|im_end|>\"\"\"

# Set stop tokens
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
        
        modelfile_path = os.path.join(model_dir, "Modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        logger.info(f"Ollama Modelfile created: {modelfile_path}")
        logger.info("To use with Ollama: ollama create my-trained-gemma3 -f " + modelfile_path)
    
    def create_deployment_script(self, model_dir: str):
        """Create a script to deploy the trained model"""
        script_content = f'''#!/bin/bash
# Deployment script for trained Gemma3 model

echo "🚀 Deploying trained Gemma3 model..."

# Start vLLM server with the trained model
python -m vllm.entrypoints.openai.api_server \\
    --model {model_dir} \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --tensor-parallel-size 8 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 4096 \\
    --served-model-name "trained-gemma3-12b" \\
    --trust-remote-code

echo "✅ Trained model deployed at http://localhost:8000"
echo "Model name: trained-gemma3-12b"
echo "Use this model name in tau-bench with: --model trained-gemma3-12b --model-provider vllm"
'''
        
        script_path = f"{model_dir}/deploy_model.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)
        logger.info(f"Deployment script created: {script_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO Training for Gemma3")
    parser.add_argument("--output_dir", type=str, default="./rl_training/outputs")
    parser.add_argument("--log_dir", type=str, default="./rl_training/logs")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--rollouts_per_episode", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--save_steps", type=int, default=5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--test_run", action="store_true", help="Run in test mode (no wandb)")
    args = parser.parse_args()
    
    # PPO Configuration
    config = PPOConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_episodes=args.num_episodes if not args.test_run else 2,
        rollouts_per_episode=args.rollouts_per_episode if not args.test_run else 5,
        learning_rate=args.learning_rate,
        critic_learning_rate=1e-4,
        ppo_epochs=4,
        clip_epsilon=0.2,
        kl_coeff=0.02,
        max_kl=0.1,
        gamma=0.99,
        gae_lambda=0.95,
        max_grad_norm=1.0,
        save_every=args.save_steps,
        use_flash_attention=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        weight_decay=0.01,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_sequence_length=args.max_sequence_length,
        use_wandb=not args.test_run,
        test_run=args.test_run
    )
    
    # Initialize trainer
    trainer = PPOTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
