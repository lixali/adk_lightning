#!/usr/bin/env python3
"""
Test available Gemma models through different providers
"""

import os
import sys

def test_huggingface_gemma():
    """Test Gemma models via HuggingFace"""
    print("🔍 Testing Gemma models via HuggingFace...")
    
    # Check if HuggingFace API key is available
    hf_token = os.environ.get("HUGGINGFACE_API_KEY") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HuggingFace API key not found. Set HUGGINGFACE_API_KEY or HF_TOKEN environment variable.")
        return []
    
    models_to_test = [
        "google/gemma-3-12b-it",
    ]
    
    available_models = []
    
    try:
        from litellm import completion
        
        for model in models_to_test:
            try:
                print(f"Testing {model}...", end=" ")
                response = completion(
                    model=f"huggingface/{model}",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10,
                    api_key=hf_token
                )
                print("✅ Available")
                available_models.append(model)
            except Exception as e:
                error_str = str(e).lower()
                if "404" in error_str or "not found" in error_str:
                    print("❌ Not found")
                elif "authorization" in error_str or "access" in error_str:
                    print("🔒 Access denied")
                elif "quota" in error_str or "limit" in error_str:
                    print("⏱️ Quota exceeded")
                else:
                    print(f"⚠️ Error: {str(e)[:100]}...")
                    
    except ImportError:
        print("❌ litellm not installed. Run: pip install litellm")
        return []
    
    return available_models

def test_ollama_gemma():
    """Test Gemma models via Ollama (local)"""
    print("\n🔍 Testing Gemma models via Ollama (local)...")
    
    try:
        import requests
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            gemma_models = [m["name"] for m in models if "gemma" in m["name"].lower()]
            
            if gemma_models:
                print(f"✅ Found {len(gemma_models)} Gemma model(s) in Ollama:")
                for model in gemma_models:
                    print(f"   - {model}")
                return gemma_models
            else:
                print("ℹ️ Ollama is running but no Gemma models found.")
                print("   To install Gemma: ollama pull gemma:7b")
                return []
        else:
            print("❌ Ollama not responding")
            return []
    except Exception as e:
        print(f"❌ Ollama not available: {e}")
        print("   Install Ollama: https://ollama.ai")
        return []

def test_gemma_integration():
    """Test our Gemma integration"""
    print("\n🔍 Testing tau-bench Gemma integration...")
    
    try:
        from tau_bench.model_utils.model.general_model import model_factory
        from tau_bench.model_utils.model.model import Platform
        
        # Test model creation
        print("Testing model factory...", end=" ")
        model = model_factory(
            model_id="gemma-2-9b-it",
            platform=Platform.GEMMA,
            temperature=0.0
        )
        print("✅ Model factory works")
        
        # Test model info
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"Model info: {info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def generate_usage_examples(hf_models, ollama_models):
    """Generate usage examples based on available models"""
    print("\n🎯 Usage Examples:")
    
    if hf_models:
        print("\n📡 HuggingFace (hosted) examples:")
        model = hf_models[0].replace("google/", "")
        print(f"# Using {model} via HuggingFace")
        print(f"python run.py --agent-strategy tool-calling --env retail \\")
        print(f"  --model {model} --model-provider gemma \\")
        print(f"  --user-model gemini-2.5-flash --user-model-provider vertex_ai \\")
        print(f"  --user-strategy llm --max-concurrency 1")
        
    if ollama_models:
        print("\n🏠 Ollama (local) examples:")
        model = ollama_models[0]
        print(f"# Using {model} via Ollama")
        print(f"python run.py --agent-strategy tool-calling --env retail \\")
        print(f"  --model {model} --model-provider gemma \\")
        print(f"  --user-model gemini-2.5-flash --user-model-provider vertex_ai \\")
        print(f"  --user-strategy llm --max-concurrency 1")
    
    print("\n🔧 Future fine-tuned model example:")
    print("# When you have a fine-tuned model")
    print("python run.py --agent-strategy tool-calling --env retail \\")
    print("  --model /path/to/your/fine-tuned-gemma \\")
    print("  --model-provider gemma \\")
    print("  --user-model gemini-2.5-flash --user-model-provider vertex_ai \\")
    print("  --user-strategy llm --max-concurrency 1")

def main():
    print("🚀 Gemma Model Integration Test")
    print("=" * 50)
    
    # Test different providers
    hf_models = test_huggingface_gemma()
    ollama_models = test_ollama_gemma()
    integration_works = test_gemma_integration()
    
    # Summary
    print("\n📊 Summary:")
    print(f"✅ HuggingFace models available: {len(hf_models)}")
    print(f"✅ Ollama models available: {len(ollama_models)}")
    print(f"✅ Integration working: {integration_works}")
    
    if hf_models or ollama_models:
        generate_usage_examples(hf_models, ollama_models)
    else:
        print("\n⚠️ No Gemma models currently available.")
        print("Setup options:")
        print("1. Set HUGGINGFACE_API_KEY for hosted models")
        print("2. Install Ollama and run: ollama pull gemma:7b")
        print("3. Set up vLLM server with Gemma models")

if __name__ == "__main__":
    main()
