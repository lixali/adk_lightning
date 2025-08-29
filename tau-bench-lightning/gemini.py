import os
import warnings
import logging

# Suppress gRPC warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Initialize the Vertex AI SDK with your project ID and location
vertexai.init(project="google.com:agent-iteration", location="us-central1")

print("=== APPROACH 1: Direct Model Usage ===")
# Create the model directly
model = GenerativeModel("gemini-2.5-flash")
# model = GenerativeModel("gemini-1.5-flash")

# Test basic generation with model
print("Basic generation with model:")
response = model.generate_content("What is the capital of France?")
print(response.text)

# print("\nStreaming generation with model:")
# # Use streaming with model
# for chunk in model.generate_content("Why is the sky blue?", stream=True):
#     print(chunk.text, end="")

# print("\n\n" + "="*50)
# print("=== APPROACH 2: Agent Usage ===")

# # Create an agent (reasoning engine)
# class SimpleAgent:
#     def __init__(self, model_name="gemini-2.5-flash"):
#         self.model = GenerativeModel(model_name)
#         self.conversation_history = []
    
#     def run_inference(self, prompt, use_history=False):
#         if use_history:
#             # Include conversation history
#             full_prompt = "\n".join(self.conversation_history + [f"User: {prompt}"])
#         else:
#             full_prompt = prompt
        
#         response = self.model.generate_content(full_prompt)
        
#         # Add to history
#         self.conversation_history.append(f"User: {prompt}")
#         self.conversation_history.append(f"Assistant: {response.text}")
        
#         return response.text
    
#     def stream_inference(self, prompt):
#         response_text = ""
#         for chunk in self.model.generate_content(prompt, stream=True):
#             print(chunk.text, end="")
#             response_text += chunk.text
#         return response_text

# # Create and use the agent
# agent = SimpleAgent("gemini-2.5-flash")

# print("Basic inference with agent:")
# result1 = agent.run_inference("What is the capital of France?")
# print(result1)

# print("\nStreaming inference with agent:")
# result2 = agent.stream_inference("Explain quantum computing in simple terms.")

# print("\n\nConversational inference with agent (using history):")
# result3 = agent.run_inference("What did I just ask you about?", use_history=True)
# print(result3)

# print("\n\nDone!")