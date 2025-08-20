from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import os

class GemmaModel():  # Inherit from BaseLlm
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return list of model patterns supported by this class."""
        return ["gemma-12b"]  # Only support 12B model

    def __init__(self, model: str = "gemma-12b"):
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

    async def generate_content_async(self, llm_request, stream: bool = False):
        """Generate content asynchronously for ADK compatibility."""
        # Import ADK response types
        from google.genai import types
        from ..models.llm_response import LlmResponse
        
        # Extract prompt from LlmRequest
        prompt = self._build_prompt_from_request(llm_request)
        
        # Generate response using the existing generate method
        response_text = self.generate(prompt)
        
        # Remove the input prompt from the response if it's included
        if response_text.startswith(prompt):
            response_text = response_text[len(prompt):].strip()
        
        # Create proper ADK response structure
        content = types.Content(
            role="model",
            parts=[types.Part.from_text(text=response_text)]
        )
        
        # Yield the response (ADK expects an async generator)
        yield LlmResponse(content=content)
    
    def _build_prompt_from_request(self, llm_request):
        """Build prompt from LlmRequest object."""
        prompt_parts = []
        
        # Add system instruction if present
        if hasattr(llm_request, 'config') and llm_request.config and hasattr(llm_request.config, 'system_instruction'):
            if llm_request.config.system_instruction:
                prompt_parts.append(f"System: {llm_request.config.system_instruction}")
        
        # Add conversation history
        if hasattr(llm_request, 'contents') and llm_request.contents:
            for content in llm_request.contents:
                role = content.role if hasattr(content, 'role') else 'user'
                if hasattr(content, 'parts') and content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            prompt_parts.append(f"{role.capitalize()}: {part.text}")
        
        return "\n".join(prompt_parts) if prompt_parts else "Hello"

# mymodel = GemmaModel()
# mymodel.generate("Hello, Gemma!")  # Example usage