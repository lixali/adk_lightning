#!/usr/bin/env python3
"""Test script to verify gemma3:12b integration"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tau_bench.model_utils.model.gemma import GemmaModel

def test_gemma3_12b():
    """Test gemma3:12b model integration"""
    
    print("Testing gemma3:12b integration...")
    
    # Initialize the model
    model = GemmaModel(
        model="gemma3:12b",
        provider="ollama",
        temperature=0.0
    )
    
    print(f"Model name: {model.model}")
    print(f"Provider: {model.provider}")
    print(f"Temperature: {model.temperature}")
    
    # Check configuration values
    print(f"\nConfiguration values:")
    print(f"Capability score: {model.get_capability()}")
    
    # Get model info which should contain other details
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    # Create a dummy datapoint to test cost and latency
    from tau_bench.model_utils.api.datapoint import GenerateDatapoint
    dummy_dp = GenerateDatapoint(
        instruction="Test instruction",
        text="Test input text"
    )
    print(f"Approx cost: {model.get_approx_cost(dummy_dp)}")
    print(f"Latency: {model.get_latency(dummy_dp)}")
    
    # Test a simple prompt
    print(f"\nTesting inference...")
    try:
        from tau_bench.model_utils.model.chat import Message, Role
        messages = [Message(role=Role.USER, content="Hello! Can you briefly introduce yourself?")]
        response = model.generate_message(messages, force_json=False)
        print(f"Response: {response.content[:100]}...")
        print("✅ gemma3:12b integration successful!")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_gemma3_12b()
    sys.exit(0 if success else 1)
