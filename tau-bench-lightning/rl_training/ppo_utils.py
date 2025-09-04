#!/usr/bin/env python3
"""
PPO Utilities and Configuration Classes
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import torch.nn.functional as F

@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    model_name: str = "google/gemma3-12b-it"
    output_dir: str = "./rl_training/outputs"
    num_episodes: int = 50
    rollouts_per_episode: int = 20
    learning_rate: float = 1e-5
    critic_learning_rate: float = 1e-4
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.02
    max_kl: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    save_every: int = 5
    use_flash_attention: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    weight_decay: float = 0.01
    temperature: float = 0.7
    max_length: int = 2048
    gradient_accumulation_steps: int = 1
    max_sequence_length: int = 1024
    use_wandb: bool = True
    test_run: bool = False

class PPOMemory:
    """Memory buffer for storing PPO training data"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def clear(self):
        """Clear all stored data"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def push(self, state, action, reward, log_prob, value, done):
        """Add new experience to memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def get_batches(self, batch_size: int):
        """Get batches for training"""
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            
            yield {
                'states': [self.states[i] for i in batch_indices],
                'actions': [self.actions[i] for i in batch_indices],
                'rewards': [self.rewards[i] for i in batch_indices],
                'log_probs': [self.log_probs[i] for i in batch_indices],
                'values': [self.values[i] for i in batch_indices],
                'dones': [self.dones[i] for i in batch_indices]
            }

def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)
    
    Args:
        rewards: Tensor of rewards
        values: Tensor of value function estimates
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    
    Returns:
        advantages: Computed advantages
        returns: Computed returns (advantages + values)
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    # Compute advantages using GAE
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * gae_lambda * last_advantage
        last_advantage = advantages[t]
    
    returns = advantages + values
    
    return advantages, returns

def compute_kl_divergence(log_probs_old: torch.Tensor, log_probs_new: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between old and new policy"""
    return (log_probs_old.exp() * (log_probs_old - log_probs_new)).sum(dim=-1).mean()

def compute_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of policy distribution"""
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1).mean()

class AdaptiveKLController:
    """Adaptive KL divergence controller for PPO"""
    
    def __init__(self, init_kl_coeff: float = 0.02, target_kl: float = 0.01, horizon: int = 10000):
        self.kl_coeff = init_kl_coeff
        self.target_kl = target_kl
        self.horizon = horizon
    
    def update(self, current_kl: float) -> float:
        """Update KL coefficient based on current KL divergence"""
        if current_kl < self.target_kl / 1.5:
            self.kl_coeff /= 2
        elif current_kl > self.target_kl * 1.5:
            self.kl_coeff *= 2
        
        return self.kl_coeff

def whiten_rewards(rewards: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Whiten rewards by normalizing"""
    if shift_mean:
        rewards = rewards - rewards.mean()
    
    return rewards / (rewards.std() + 1e-8)

def prepare_model_inputs(tokenizer, texts: List[str], max_length: int = 2048):
    """Prepare model inputs from text"""
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return inputs

def generate_with_model(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7, do_sample: bool = True):
    """Generate text with the model"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text, outputs[0]

def calculate_sequence_log_probs(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
    """Calculate log probabilities for a sequence"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Shift for causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs for actual tokens
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Handle padding tokens
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            token_log_probs = token_log_probs * shift_attention_mask
        
        return token_log_probs.sum(dim=-1)  # Sum over sequence length

class PPOLoss:
    """PPO loss computation"""
    
    def __init__(self, clip_epsilon: float = 0.2, value_loss_coeff: float = 0.5, entropy_coeff: float = 0.01):
        self.clip_epsilon = clip_epsilon
        self.value_loss_coeff = value_loss_coeff
        self.entropy_coeff = entropy_coeff
    
    def compute_policy_loss(self, log_probs_old: torch.Tensor, log_probs_new: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Compute clipped policy loss"""
        ratio = torch.exp(log_probs_new - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        return -torch.min(surr1, surr2).mean()
    
    def compute_value_loss(self, values_pred: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Compute value function loss"""
        return F.mse_loss(values_pred, returns)
    
    def compute_entropy_loss(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy loss for exploration"""
        return compute_entropy(log_probs)
    
    def compute_total_loss(self, policy_loss: torch.Tensor, value_loss: torch.Tensor, entropy_loss: torch.Tensor) -> torch.Tensor:
        """Compute total PPO loss"""
        return policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy_loss
