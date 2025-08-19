from google.adk.models.gemma_model import GemmaModel
from google.adk.model_handlers.base_handler import BaseModelHandler

class GemmaModelHandler(BaseModelHandler):
    def __init__(self):
        super().__init__()
        self.model = GemmaModel()
        
    def generate_content(self, prompt, **kwargs):
        return self.model.generate(prompt, **kwargs)
    
    def stream_content(self, prompt, **kwargs):
        # For streaming implementation
        response = self.generate_content(prompt, **kwargs)
        yield response