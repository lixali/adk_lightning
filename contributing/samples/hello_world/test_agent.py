#!/usr/bin/env python3

"""
Test script to properly run the hello_world agent with function calling.
This will test the calculate_math_expression function.
"""

import asyncio
import logging
from google.adk.runners import InMemoryRunner
from google.genai import types
from agent import root_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_math_function():
    """Test the math calculation function."""
    
    print("=" * 60)
    print("Testing Math Function in Hello World Agent")
    print("=" * 60)
    
    # Create runner
    runner = InMemoryRunner(
        agent=root_agent,
        app_name='hello_world_test'
    )
    
    # Create session
    session = await runner.session_service.create_session(
        app_name='hello_world_test',
        user_id='test_user'
    )
    
    # Test cases
    test_cases = [
        "Calculate 123432 divided by 3.3",
        "What is 2 + 3 * 4?",
        "Find the square root of 16",
        "What's 15% of 200?",
        "Compute sin(pi/2)",
        "Calculate factorial(5)"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_input} ---")
        
        content = types.Content(
            role='user',
            parts=[types.Part.from_text(text=test_input)]
        )
        
        try:
            async for event in runner.run_async(
                user_id='test_user',
                session_id=session.id,
                new_message=content,
            ):
                if event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(f"Agent: {part.text}")
                        elif part.function_call:
                            print(f"Function Call: {part.function_call.name}({part.function_call.args})")
                        elif part.function_response:
                            print(f"Function Response: {part.function_response.response}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    asyncio.run(test_math_function())
