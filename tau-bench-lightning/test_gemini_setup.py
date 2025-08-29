#!/usr/bin/env python3
"""
Test script to verify Gemini/Vertex AI authentication and basic functionality
"""

import os
import sys

def test_environment():
    """Test if required environment variables are set"""
    print("=== Environment Variables Check ===")
    
    # Check for Google authentication
    google_creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    google_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {'✓' if google_creds else '✗'}")
    print(f"GOOGLE_API_KEY: {'✓' if google_api_key else '✗'}")
    print(f"GOOGLE_CLOUD_PROJECT: {'✓' if google_project else '✗'}")
    
    if not any([google_creds, google_api_key]):
        print("\n❌ No Google authentication found!")
        print("Please set one of:")
        print("  - GOOGLE_APPLICATION_CREDENTIALS (for service account)")
        print("  - GOOGLE_API_KEY (for API key)")
        return False
    
    return True

def test_litellm_providers():
    """Test available providers in litellm"""
    print("\n=== LiteLLM Providers Check ===")
    try:
        from litellm import provider_list, LlmProviders
        
        google_providers = []
        for provider in provider_list:
            if 'vertex' in str(provider).lower() or 'gemini' in str(provider).lower():
                google_providers.append(provider.value)
        
        print(f"Available Google providers: {google_providers}")
        return len(google_providers) > 0
    except Exception as e:
        print(f"❌ Error checking providers: {e}")
        return False

def test_simple_completion():
    """Test a simple completion with Gemini"""
    print("\n=== Simple Completion Test ===")
    
    try:
        from litellm import completion
        import warnings
        warnings.filterwarnings('ignore')
        
        # Try vertex_ai provider first
        print("Testing vertex_ai provider...")
        response = completion(
            model="gemini-2.5-flash",
            custom_llm_provider="vertex_ai",
            messages=[{"role": "user", "content": "Say 'Hello from Gemini!'"}],
            timeout=10
        )
        print(f"✓ vertex_ai works: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ vertex_ai failed: {e}")
        
        # Try gemini provider as fallback
        try:
            print("Testing gemini provider...")
            response = completion(
                model="gemini-2.5-flash",
                custom_llm_provider="gemini",
                messages=[{"role": "user", "content": "Say 'Hello from Gemini!'"}],
                timeout=10
            )
            print(f"✓ gemini works: {response.choices[0].message.content}")
            return True
        except Exception as e2:
            print(f"❌ gemini failed: {e2}")
            return False

def main():
    print("🧪 Testing Gemini/Vertex AI Integration for tau-bench\n")
    
    # Run tests
    env_ok = test_environment()
    providers_ok = test_litellm_providers()
    completion_ok = test_simple_completion()
    
    print("\n=== Summary ===")
    print(f"Environment: {'✓' if env_ok else '✗'}")
    print(f"Providers: {'✓' if providers_ok else '✗'}")
    print(f"Completion: {'✓' if completion_ok else '✗'}")
    
    if all([env_ok, providers_ok, completion_ok]):
        print("\n🎉 All tests passed! Gemini integration should work.")
        print("\nYou can now run:")
        print("python run.py --agent-strategy tool-calling --env retail --model gemini-2.5-flash --model-provider vertex_ai --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1")
    else:
        print("\n❌ Some tests failed. Please check the setup.")
        
        if not env_ok:
            print("\n🔧 To fix authentication:")
            print("1. For API Key: export GOOGLE_API_KEY='your-api-key'")
            print("2. For Service Account: export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'")
            print("3. Make sure GOOGLE_CLOUD_PROJECT is set if using service account")

if __name__ == "__main__":
    main()
