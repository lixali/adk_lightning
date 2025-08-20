from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
import os
import logging

logger = logging.getLogger(__name__)

class GemmaModel():  # Inherit from BaseLlm
    # Class-level cache to store the loaded model and tokenizer
    _model_cache = {}
    _tokenizer_cache = {}
    _initialized_models = set()
    _login_completed = False
    
    @classmethod
    def supported_models(cls) -> list[str]:
        """Return list of model patterns supported by this class."""
        return ["gemma-12b"]  # Only support 12B model

    def __init__(self, model: str = "gemma-12b"):
        # Only support 12B model
        self.model_name = "google/gemma-3-12b-it"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"GemmaModel.__init__ called for {self.model_name}")
        
        # Check if this model is already loaded in cache
        if self.model_name in self._model_cache:
            logger.info(f"✅ Using cached Gemma 12B model on {self.device} - avoiding reload!")
            print(f"✅ Using cached Gemma 12B model on {self.device} - avoiding reload!")
            self.model = self._model_cache[self.model_name]
            self.tokenizer = self._tokenizer_cache[self.model_name]
            return
        
        # Ensure you have HUGGING_FACE_TOKEN in environment
        hf_token = os.getenv("HUGGING_FACE_TOKEN")
        if not hf_token:
            raise ValueError("Please set HUGGING_FACE_TOKEN environment variable")
            
        # Login to Hugging Face (only once)
        if not self._login_completed:
            logger.info("🔐 Performing HuggingFace login (one-time)")
            login(token=hf_token)
            self._login_completed = True
        
        logger.info(f"🚀 Loading Gemma 12B model on {self.device} (first time load)...")
        print(f"🚀 Loading Gemma 12B model on {self.device} (first time load)...")
        
        # Initialize tokenizer and model
        logger.info("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=hf_token
        )
        
        logger.info("🧠 Loading model weights...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=hf_token,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Cache the loaded model and tokenizer
        self._model_cache[self.model_name] = self.model
        self._tokenizer_cache[self.model_name] = self.tokenizer
        
        logger.info(f"✅ Model loaded successfully on {self.device} and cached for future use!")
        print(f"✅ Model loaded successfully on {self.device} and cached for future use!")
        
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
        
        # Clean up any role prefixes that might appear in the response
        response_text = self._clean_response_text(response_text)
        
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
        
        # Add system instruction if present (without "System:" prefix)
        if hasattr(llm_request, 'config') and llm_request.config and hasattr(llm_request.config, 'system_instruction'):
            if llm_request.config.system_instruction:
                prompt_parts.append(llm_request.config.system_instruction)
        
        # Add conversation history (without role prefixes to avoid echoing)
        if hasattr(llm_request, 'contents') and llm_request.contents:
            for content in llm_request.contents:
                if hasattr(content, 'parts') and content.parts:
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            # Only add the actual text without role prefixes
                            prompt_parts.append(part.text)
        
        return "\n".join(prompt_parts) if prompt_parts else "Hello"
    
    def _clean_response_text(self, response_text):
        """Clean up response text by removing unwanted prefixes and formatting."""
        if not response_text:
            return response_text
        
        # Remove common role prefixes that might appear at the start of responses
        prefixes_to_remove = [
            "Model: ",
            "model: ",
            "System: ",
            "system: ",
            "Assistant: ",
            "assistant: ",
            "AI: ",
            "ai: "
        ]
        
        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
                break
        
        # Remove any leading/trailing whitespace
        response_text = response_text.strip()
        
        return response_text

# mymodel = GemmaModel()
# mymodel.generate("Hello, Gemma!")  # Example usage