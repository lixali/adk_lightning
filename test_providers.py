#!/usr/bin/env python3
"""
Test specific provider combinations for Gemini
"""

import os
import sys

def test_provider(provider_name):
    """Test a specific provider"""
    print(f"\n=== Testing {provider_name} provider ===")
    
    try:
        from litellm import completion
        import warnings
        warnings.filterwarnings('ignore')
        
        response = completion(
            model="gemini-2.5-flash",
            custom_llm_provider=provider_name,
            messages=[{"role": "user", "content": "What is 2+2? Just answer with the number."}],
            timeout=15
        )
        result = response.choices[0].message.content.strip()
        print(f"✓ {provider_name} works: {result}")
        return True
        
    except Exception as e:
        print(f"❌ {provider_name} failed: {e}")
        return False

def test_tool_calling():
    """Test tool calling functionality"""
    print(f"\n=== Testing Tool Calling ===")
    
    try:
        from litellm import completion
        import warnings
        warnings.filterwarnings('ignore')
        
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }]
        
        response = completion(
            model="gemini-2.5-flash",
            custom_llm_provider="vertex_ai",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            timeout=15
        )
        
        if response.choices[0].message.tool_calls:
            print("✓ Tool calling works!")
            return True
        else:
            print("❌ No tool calls in response")
            return False
        
    except Exception as e:
        print(f"❌ Tool calling failed: {e}")
        return False

def main():
    print("🧪 Testing Gemini Provider Combinations\n")
    
    # Test different providers
    providers_to_test = ["vertex_ai", "gemini"]
    working_providers = []
    
    for provider in providers_to_test:
        if test_provider(provider):
            working_providers.append(provider)
    
    # Test tool calling if any provider works
    if working_providers:
        test_tool_calling()
    
    print(f"\n=== Summary ===")
    print(f"Working providers: {working_providers}")
    
    if working_providers:
        print(f"\n🎉 Recommended command:")
        provider = working_providers[0]
        print(f"python run.py --agent-strategy tool-calling --env retail --model gemini-2.5-flash --model-provider {provider} --user-model gemini-2.5-flash --user-model-provider {provider} --user-strategy llm --max-concurrency 1")
    else:
        print("\n❌ No working providers found")

if __name__ == "__main__":
    main()
