from google.adk.model_handlers.gemma_handler import GemmaModelHandler
from google.adk import constants

def initialize_models():
    """Initialize and return model handlers"""
    model_handlers = {}
    
    # Initialize Gemma 12B
    gemma_handler = GemmaModelHandler()
    model_handlers[constants.GEMMA_12B] = gemma_handler
    
    return model_handlers

if __name__ == "__main__":
    handlers = initialize_models()
    default_model = handlers[constants.DEFAULT_MODEL]
    
    # Test generation
    response = default_model.generate_content(
        "Write a Python function to calculate fibonacci sequence"
    )
    print(response)