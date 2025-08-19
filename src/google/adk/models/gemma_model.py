from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import os
from .base_llm import BaseLlm  # Add this import

class GemmaModel(BaseLlm):  # Inherit from BaseLlm
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return list of model patterns supported by this class."""
        return ["gemma-12b"]  # Only support 12B model

    def __init__(self):
        # Ensure you have HUGGING_FACE_TOKEN in environment
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("Please set HUGGING_FACE_TOKEN environment variable")
            
        # Login to Hugging Face
        login(token=hf_token)
        
        # Only support 12B model
        self.model_name = "google/gemma-3-12b-it"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Gemma 12B model on {self.device}...")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print(f"Model loaded successfully on {self.device}!")
        
    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        print(f"Generating content for prompt: {prompt}")
        print(f"inputs: {inputs}")
        outputs = self.model.generate(
            **inputs,
            max_length=kwargs.get("max_tokens", 8192),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.95),
            do_sample=True
        )

        print(f"Generated outputs: {outputs}")
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
