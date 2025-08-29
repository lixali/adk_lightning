import os
from tau_bench.model_utils.api.datapoint import Datapoint
from tau_bench.model_utils.model.chat import ChatModel, Message
from tau_bench.model_utils.model.completion import approx_cost_for_datapoint, approx_prompt_str
from tau_bench.model_utils.model.general_model import wrap_temperature
from tau_bench.model_utils.model.utils import approx_num_tokens

DEFAULT_GOOGLE_MODEL = "gemini-2.5-flash"
API_KEY_ENV_VAR = "GOOGLE_API_KEY"

PRICE_PER_INPUT_TOKEN_MAP = {
    "gemini-2.5-flash": 0.075 / 1000000,  # $0.075 per 1M input tokens
    "gemini-1.5-flash": 0.075 / 1000000,  # $0.075 per 1M input tokens
    "gemini-1.5-flash-001": 0.075 / 1000000,
    "gemini-1.5-flash-002": 0.075 / 1000000,
    "gemini-1.5-pro": 1.25 / 1000000,     # $1.25 per 1M input tokens
    "gemini-1.5-pro-001": 1.25 / 1000000,
    "gemini-1.5-pro-002": 1.25 / 1000000,
}

PRICE_PER_OUTPUT_TOKEN_MAP = {
    "gemini-2.5-flash": 0.3 / 1000000,    # $0.3 per 1M output tokens
    "gemini-1.5-flash": 0.3 / 1000000,    # $0.3 per 1M output tokens
    "gemini-1.5-flash-001": 0.3 / 1000000,
    "gemini-1.5-flash-002": 0.3 / 1000000,
    "gemini-1.5-pro": 5.0 / 1000000,      # $5.0 per 1M output tokens
    "gemini-1.5-pro-001": 5.0 / 1000000,
    "gemini-1.5-pro-002": 5.0 / 1000000,
}

INPUT_PRICE_PER_TOKEN_FALLBACK = 1.0 / 1000000
OUTPUT_PRICE_PER_TOKEN_FALLBACK = 4.0 / 1000000

CAPABILITY_SCORE_MAP = {
    "gemini-2.5-flash": 0.9,
    "gemini-1.5-flash": 0.8,
    "gemini-1.5-flash-001": 0.8,
    "gemini-1.5-flash-002": 0.8,
    "gemini-1.5-pro": 0.95,
    "gemini-1.5-pro-001": 0.95,
    "gemini-1.5-pro-002": 0.95,
}

CAPABILITY_SCORE_FALLBACK = 0.8

# TODO: implement proper latency measurements
LATENCY_MS_PER_OUTPUT_TOKEN_MAP = {}
LATENCY_MS_PER_OUTPUT_TOKEN_FALLBACK = 50.0

MAX_CONTEXT_LENGTH_MAP = {
    "gemini-2.5-flash": 1048576,  # 1M tokens
    "gemini-1.5-flash": 1048576,  # 1M tokens
    "gemini-1.5-flash-001": 1048576,
    "gemini-1.5-flash-002": 1048576,
    "gemini-1.5-pro": 2097152,   # 2M tokens
    "gemini-1.5-pro-001": 2097152,
    "gemini-1.5-pro-002": 2097152,
}

MAX_CONTEXT_LENGTH_FALLBACK = 1048576


class GoogleModel(ChatModel):
    def __init__(
        self,
        model: str = DEFAULT_GOOGLE_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
    ):
        if api_key is None:
            api_key = os.environ.get(API_KEY_ENV_VAR)
        if api_key is None:
            raise ValueError(
                f"API key is required for Google models. "
                f"Set the {API_KEY_ENV_VAR} environment variable or pass it as an argument."
            )
        
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    def generate_from_messages(
        self, messages: list[Message], temperature: float | None = None
    ) -> str:
        from litellm import completion
        
        if temperature is None:
            temperature = self.temperature
        
        # Convert our Message objects to dict format for litellm
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "role": message.role.value,
                "content": message.content
            })
        
        response = completion(
            model=self.model,
            custom_llm_provider="vertex_ai",  # Use vertex_ai provider for better integration
            messages=formatted_messages,
            temperature=wrap_temperature(temperature),
            api_key=self.api_key,
        )
        
        return response.choices[0].message.content

    def get_approx_cost(self, dp: Datapoint) -> float:
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
