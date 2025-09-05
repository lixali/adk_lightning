"""
SFT Model Provider for tau-bench integration.

This module provides a model provider that can load and use SFT fine-tuned models
in the tau-bench evaluation framework. It's compatible with the existing model provider
architecture and can be used as a drop-in replacement for other providers.

Usage:
    python run.py --agent-strategy tool-calling --env retail \
        --model sft-gemma3-12b --model-provider sft \
        --user-model gemini-2.5-pro --user-model-provider vertex_ai \
        --user-strategy llm --max-concurrency 5
"""

import os
import torch
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class SFTModelProvider:
    """
    Model provider for SFT fine-tuned models.
    
    This provider loads locally fine-tuned models and provides an interface
    compatible with the tau-bench model provider system.
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the SFT model provider.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to load the model on ("auto", "cuda", "cpu")
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for SFT model provider")
        
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading SFT model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("SFT model loaded successfully")
    
    def completion(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion using the fine-tuned model.
        
        Args:
            messages: List of conversation messages
            tools: List of available tools (for function calling)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments
            
        Returns:
            Dict containing the completion response in OpenAI format
        """
        try:
            # Format the conversation using chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            )
            
            # Move to device
            if torch.cuda.is_available() and self.device != "cpu":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode the generated response
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Parse potential tool calls from the generated text
            tool_calls = self._parse_tool_calls(generated_text)
            
            # Format response in OpenAI-compatible format
            response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": generated_text.strip(),
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": inputs['input_ids'].shape[1],
                    "completion_tokens": len(outputs[0]) - inputs['input_ids'].shape[1],
                    "total_tokens": len(outputs[0])
                }
            }
            
            # Add tool calls if detected
            if tool_calls:
                response["choices"][0]["message"]["tool_calls"] = tool_calls
                response["choices"][0]["finish_reason"] = "tool_calls"
            
            return response
            
        except Exception as e:
            logger.error(f"Error in SFT model completion: {e}")
            raise
    
    def _parse_tool_calls(self, generated_text: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from generated text.
        
        This is a simplified parser - in practice, you might want to train
        the model to generate tool calls in a specific format.
        
        Args:
            generated_text: The generated response text
            
        Returns:
            List of tool call dictionaries
        """
        import json
        import re
        
        tool_calls = []
        
        # Look for JSON-like tool call patterns
        # The SFT model should learn to generate these from training data
        tool_call_pattern = r'\{[^}]*"name"[^}]*"arguments"[^}]*\}'
        matches = re.findall(tool_call_pattern, generated_text)
        
        for i, match in enumerate(matches):
            try:
                # Try to parse as JSON
                parsed = json.loads(match)
                if "name" in parsed and "arguments" in parsed:
                    tool_call = {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": parsed["name"],
                            "arguments": json.dumps(parsed["arguments"])
                        }
                    }
                    tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        
        return tool_calls

# Integration function for tau-bench
def get_sft_model_provider(model_name: str, **kwargs) -> SFTModelProvider:
    """
    Factory function to create SFT model provider.
    
    Args:
        model_name: Model identifier (e.g., "sft-gemma3-12b")
        **kwargs: Additional arguments
        
    Returns:
        SFTModelProvider instance
    """
    # Map model names to actual paths
    model_paths = {
        "sft-gemma3-12b": "sft_training/checkpoints/final_model",
        # Add more mappings as needed
    }
    
    model_path = model_paths.get(model_name)
    if not model_path:
        raise ValueError(f"Unknown SFT model: {model_name}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SFT model not found at: {model_path}")
    
    return SFTModelProvider(model_path, **kwargs)

# Function to integrate with existing model provider system
def create_completion(model, messages, tools=None, **kwargs):
    """
    Compatibility function for existing tau-bench model provider interface.
    """
    provider = get_sft_model_provider(model)
    return provider.completion(messages, tools=tools, **kwargs)
