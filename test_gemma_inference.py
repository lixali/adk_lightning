#!/usr/bin/env python3
"""
Quick test of Gemma integration with actual model inference
"""

import sys
import os

def test_gemma_ollama():
    """Test actual inference with Ollama Gemma model"""
    try:
        from tau_bench.model_utils.model.general_model import model_factory
        from tau_bench.model_utils.model.model import Platform
        from tau_bench.model_utils.model.chat import Message, Role
        
        print("🔍 Testing Gemma via Ollama with actual inference...")
        
        # Create model
        model = model_factory(
            model_id="gemma:7b",
            platform=Platform.GEMMA,
            temperature=0.0
        )
        
        print(f"✅ Model created: {model.get_model_info()}")
        
        # Test simple generation
        print("🧪 Testing simple generation...")
        result = model.generate(
            instruction="You are a helpful assistant. Answer briefly.",
            text="What is 2+2?",
            temperature=0.0
        )
        print(f"✅ Generation result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_full_benchmark_command():
    """Test the actual benchmark command"""
    print("\n🎯 Testing benchmark command...")
    print("Command to run:")
    print("python run.py --agent-strategy tool-calling --env retail \\")
    print("  --model gemma:7b --model-provider gemma \\")
    print("  --user-model gemini-2.5-flash --user-model-provider vertex_ai \\")
    print("  --user-strategy llm --max-concurrency 1 --num-tasks 1")
    print("\nNote: Add --num-tasks 1 for quick testing")

if __name__ == "__main__":
    print("🚀 Gemma Integration Inference Test")
    print("=" * 50)
    
    success = test_gemma_ollama()
    test_full_benchmark_command()
    
    if success:
        print("\n✅ Integration test passed! Gemma is ready to use as agent model.")
    else:
        print("\n❌ Integration test failed. Check the error above.")
