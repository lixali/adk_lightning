import os
from typing import Literal
from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens

DEFAULT_GEMMA_MODEL = "gemma-2-27b-it"
HUGGINGFACE_API_KEY_ENV_VAR = "HUGGINGFACE_API_KEY"

# Pricing estimates for different Gemma models (rough estimates)
PRICE_PER_INPUT_TOKEN_MAP = {
    "gemma3:12b": 0.3 / 1000000,        # Estimate for Gemma3 12B
    "gemma-2-27b-it": 0.5 / 1000000,    # Estimate for large model
    "gemma-2-9b-it": 0.2 / 1000000,     # Estimate for medium model
    "gemma-2-2b-it": 0.05 / 1000000,    # Estimate for small model
    "gemma:7b": 0.15 / 1000000,         # Original Gemma 7B
    "gemma-7b-it": 0.15 / 1000000,      # Legacy Gemma model
}

PRICE_PER_OUTPUT_TOKEN_MAP = {
    "gemma3:12b": 0.9 / 1000000,        # Estimate for Gemma3 12B
    "gemma-2-27b-it": 1.5 / 1000000,
    "gemma-2-9b-it": 0.6 / 1000000,
    "gemma-2-2b-it": 0.15 / 1000000,
    "gemma:7b": 0.45 / 1000000,         # Original Gemma 7B
    "gemma-7b-it": 0.45 / 1000000,
}

INPUT_PRICE_PER_TOKEN_FALLBACK = 0.2 / 1000000
OUTPUT_PRICE_PER_TOKEN_FALLBACK = 0.6 / 1000000

CAPABILITY_SCORE_MAP = {
    "gemma3:12b": 0.88,         # High capability for Gemma3 12B
    "gemma-2-27b-it": 0.85,     # High capability for 27B model
    "gemma-2-9b-it": 0.75,      # Good capability for 9B model  
    "gemma-2-2b-it": 0.65,      # Decent capability for 2B model
    "gemma:7b": 0.70,           # Original Gemma 7B capability
    "gemma-7b-it": 0.70,        # Legacy model capability
}

CAPABILITY_SCORE_FALLBACK = 0.75

LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {
    "gemma3:12b": 35.0,         # Estimated latency for Gemma3 12B
    "gemma-2-27b-it": 45.0,     # Higher latency for 27B model
    "gemma-2-9b-it": 20.0,      # Moderate latency for 9B model
    "gemma-2-2b-it": 10.0,      # Lower latency for 2B model
    "gemma:7b": 25.0,           # Original Gemma 7B latency
    "gemma-7b-it": 25.0,        # Legacy model latency
}
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 60.0

MAX_CONTEXT_LENGTH_MAP = {
    "gemma3:12b": 131072,       # 128k context for Gemma3 12B
    "gemma-2-27b-it": 8192,     # 8k context for 27B model
    "gemma-2-9b-it": 8192,      # 8k context for 9B model  
    "gemma-2-2b-it": 8192,      # 8k context for 2B model
    "gemma:7b": 8192,           # 8k context for original Gemma 7B
    "gemma-7b-it": 8192,        # 8k context for legacy model
}

MAX_CONTEXT_LENGTH_FALLBACK = 8192

# Provider types for flexibility
GemmaProvider = Literal["huggingface", "ollama", "vllm", "custom"]


class GemmaModel(ChatModel):
    def __init__(
        self,
        model: str = DEFAULT_GEMMA_MODEL,
        provider: GemmaProvider = "huggingface",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        model_path: str | None = None,  # For custom fine-tuned models
    ):
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.model_path = model_path  # Store path for future fine-tuned models
        self.base_url = base_url
        
        # Handle API key based on provider
        if provider == "huggingface":
            if api_key is None:
                api_key = os.environ.get(HUGGINGFACE_API_KEY_ENV_VAR)
            if api_key is None:
                raise ValueError(
                    f"API key is required for HuggingFace models. "
                    f"Set the {HUGGINGFACE_API_KEY_ENV_VAR} environment variable or pass it as an argument."
                )
        elif provider == "custom" and base_url is None:
            raise ValueError("base_url must be provided for custom Gemma models")
            
        self.api_key = api_key

    def _get_model_identifier(self) -> str:
        """Get the actual model identifier for API calls"""
        if self.model_path:
            # Use custom fine-tuned model path if provided
            return self.model_path
        elif self.provider == "huggingface":
            # Map to HuggingFace model names
            hf_model_map = {
                "gemma-2-27b-it": "google/gemma-2-27b-it",
                "gemma-2-9b-it": "google/gemma-2-9b-it", 
                "gemma-2-2b-it": "google/gemma-2-2b-it",
                "gemma-7b-it": "google/gemma-7b-it",
            }
            return hf_model_map.get(self.model, f"google/{self.model}")
        else:
            return self.model

    def generate_message(
        self,
        messages: list[Message],
        force_json: bool,
        temperature: float | None = None,
    ) -> Message:
        if temperature is None:
            temperature = self.temperature
            
        # Use the base class method to convert Messages to dict format
        msgs = self.build_generate_message_state(messages)
        
        # Get the raw content using our provider-specific method
        content = self._generate_with_provider(msgs, temperature)
        
        # Use the base class method to handle the response
        return self.handle_generate_message_response(
            prompt=msgs, content=content, force_json=force_json
        )

    def _generate_with_provider(self, messages: list[dict], temperature: float) -> str:
        """Generate content using the configured provider"""
        model_id = self._get_model_identifier()
        
        if self.provider == "huggingface":
            return self._generate_huggingface(messages, temperature, model_id)
        elif self.provider == "ollama":
            return self._generate_ollama(messages, temperature, model_id)
        elif self.provider == "vllm":
            return self._generate_vllm(messages, temperature, model_id)
        elif self.provider == "custom":
            return self._generate_custom(messages, temperature, model_id)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def supports_dp(self, dp: Datapoint) -> bool:
        """Check if this model can handle the given datapoint"""
        from tau_bench.model_utils.model.completion import approx_prompt_str
        from tau_bench.model_utils.model.utils import approx_num_tokens
        
        prompt = approx_prompt_str(dp)
        num_tokens = approx_num_tokens(prompt)
        max_context = MAX_CONTEXT_LENGTH_MAP.get(self.model, MAX_CONTEXT_LENGTH_FALLBACK)
        
        # Leave some room for the response
        return num_tokens < (max_context * 0.8)

    def _generate_huggingface(self, messages: list[dict], temperature: float, model_id: str) -> str:
        """Generate using HuggingFace Inference API via litellm"""
        from litellm import completion
        
        response = completion(
            model=f"huggingface/{model_id}",
            messages=messages,
            temperature=wrap_temperature(temperature),
            api_key=self.api_key,
        )
        return response.choices[0].message.content

    def _generate_ollama(self, messages: list[dict], temperature: float, model_id: str) -> str:
        """Generate using Ollama local deployment"""
        from litellm import completion
        
        response = completion(
            model=f"ollama/{model_id}",
            messages=messages,
            temperature=wrap_temperature(temperature),
            api_base=self.base_url or "http://localhost:11434",
        )
        return response.choices[0].message.content

    def _generate_vllm(self, messages: list[dict], temperature: float, model_id: str) -> str:
        """Generate using vLLM deployment"""
        from litellm import completion
        
        if not self.base_url:
            raise ValueError("base_url is required for vLLM provider")
            
        response = completion(
            model=model_id,
            messages=messages,
            temperature=wrap_temperature(temperature),
            api_base=self.base_url,
            api_key=self.api_key or "sk-no-api-key-required",
        )
        return response.choices[0].message.content

    def _generate_custom(self, messages: list[dict], temperature: float, model_id: str) -> str:
        """Generate using custom endpoint (for future fine-tuned models)"""
        from litellm import completion
        
        if not self.base_url:
            raise ValueError("base_url is required for custom provider")
            
        response = completion(
            model=model_id,
            messages=messages,
            temperature=wrap_temperature(temperature),
            api_base=self.base_url,
            api_key=self.api_key or "sk-no-api-key-required",
            custom_llm_provider="openai",  # Use OpenAI-compatible format for custom endpoints
        )
        return response.choices[0].message.content

    def get_approx_cost(self, dp: Datapoint) -> float:
        # For local/custom deployments, cost might be different or zero
        if self.provider in ["ollama", "custom"] and not self.model_path:
            return 0.0  # Local deployment typically has no API costs
            
        return approx_cost_for_datapoint(
            dp=dp,
            input_price_per_token=PRICE_PER_INPUT_TOKEN_MAP.get(
                self.model, INPUT_PRICE_PER_TOKEN_FALLBACK
            ),
            output_price_per_token=PRICE_PER_OUTPUT_TOKEN_MAP.get(
                self.model, OUTPUT_PRICE_PER_TOKEN_FALLBACK
            ),
            prompt_str=approx_prompt_str(dp),
        )

    def get_latency(self, dp: Datapoint) -> float:
        return LATENCY_MS_PER_OUTPUT_TOKEN_MAP.get(
            self.model, LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK
        ) * approx_num_tokens(approx_prompt_str(dp))

    def get_capability(self) -> float:
        return CAPABILITY_SCORE_MAP.get(self.model, CAPABILITY_SCORE_FALLBACK)

    def update_model_path(self, new_model_path: str):
        """Update the model path for switching to fine-tuned models"""
        self.model_path = new_model_path
        
    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""
        return {
            "model": self.model,
            "provider": self.provider,
            "model_path": self.model_path,
            "base_url": self.base_url,
            "effective_model_id": self._get_model_identifier()
        }
