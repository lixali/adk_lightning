#!/usr/bin/env python3
"""
Test available Gemini models in Vertex AI
"""

from litellm import completion
import sys

models_to_test = [
    # Gemini 2.5 variants
    "gemini-2.5-flash",
    "gemini-2.5-flash-001", 
    "gemini-2.5-flash-002",
    "gemini-2.5-pro",
    "gemini-2.5-pro-001",
    "gemini-2.5-pro-002",
    
    # Gemini 1.5 variants
    "gemini-1.5-flash",
    "gemini-1.5-flash-001", 
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.5-pro-001",
    "gemini-1.5-pro-002",
    
    # Gemini 1.0 variants
    "gemini-1.0-pro",
    "gemini-1.0-pro-001",
    "gemini-1.0-pro-002",
    "gemini-pro",
    "gemini-pro-vision",
    
    # Alternative naming patterns
    "models/gemini-2.5-flash",
    "models/gemini-1.5-flash", 
    "models/gemini-1.5-pro",
    "models/gemini-pro",
    
    # Experimental/Preview versions
    "gemini-2.5-flash-exp",
    "gemini-2.5-flash-preview",
    "gemini-1.5-flash-preview",
    "gemini-1.5-pro-preview",
    
    # Latest versions (common naming patterns)
    "gemini-flash",
    "gemini-pro-latest",
    "gemini-latest",    

]

print("🔍 Testing available Gemini models in Vertex AI...")
print()

available_models = []
unavailable_models = []

for model in models_to_test:
    try:
        print(f"Testing {model}...", end=" ")
        response = completion(
            model=model,
            custom_llm_provider="vertex_ai",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("✅ Available")
        available_models.append(model)
    except Exception as e:
        error_str = str(e).lower()
        if "404" in error_str or "not found" in error_str:
            print("❌ Not found")
        elif "permission" in error_str or "access" in error_str:
            print("🔒 Access denied")
        elif "quota" in error_str or "limit" in error_str:
            print("⏱️ Quota exceeded")
        elif "region" in error_str or "location" in error_str:
            print("🌍 Region restricted")
        else:
            print(f"⚠️ Error: {str(e)[:100]}...")
        unavailable_models.append((model, str(e)))

print()
print("📊 Results Summary:")
print(f"✅ Available models ({len(available_models)}):")
for model in available_models:
    print(f"   - {model}")

print(f"\n❌ Unavailable models ({len(unavailable_models)}):")
for model_info in unavailable_models:
    if isinstance(model_info, tuple):
        model, error = model_info
        print(f"   - {model}")
    else:
        print(f"   - {model_info}")

if available_models:
    print(f"\n🎯 Recommended commands with available models:")
    for i, model in enumerate(available_models[:3]):  # Show top 3 available models
        print(f"\nOption {i+1}: {model}")
        print(f"python run.py --agent-strategy tool-calling --env retail \\")
        print(f"  --model {model} --model-provider vertex_ai \\")
        print(f"  --user-model {model} --user-model-provider vertex_ai \\")
        print(f"  --user-strategy llm --max-concurrency 1")
else:
    print(f"\n⚠️ No models available. Check your Google Cloud project access and region settings.")
