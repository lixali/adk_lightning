# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def help_option_one(tool_context: ToolContext) -> str:
    """First help option - placeholder function.
    
    Args:
        tool_context: The tool context for this invocation.
    
    Returns:
        A helpful response offering assistance.
    """
    return "I can help you with various tasks. What else can I help you with today?"


def help_option_two(tool_context: ToolContext) -> str:
    """Second help option - placeholder function.
    
    Args:
        tool_context: The tool context for this invocation.
    
    Returns:
        Another helpful response offering assistance.
    """
    return "I'm here to assist you with any questions or tasks. What else would you like to explore?"


def calculate_math_expression(expression: str, tool_context: ToolContext) -> str:
    """Calculate a mathematical expression and return the result.
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "1245/3.3")
        tool_context: The tool context for this invocation.
    
    Returns:
        The calculated result as a string.
    """
    try:
        # Safely evaluate the mathematical expression
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


def escalate_to_representative(reason: str, tool_context: ToolContext) -> str:
    """Escalate the conversation to a live representative when the AI cannot handle the request.
    
    Args:
        reason: The reason why the AI cannot handle this question
        tool_context: The tool context for this invocation.
    
    Returns:
        A message indicating escalation to a live representative.
    """
    return f"I cannot handle this question. Let me find a live representative to help you. Reason: {reason}"


root_agent = Agent(
    model='gemma-12b',
    name='hello_world_agent',
    description='A helpful conversational AI assistant with helpful tools.',
    instruction="""You are a helpful AI assistant. 
    Answer questions clearly and concisely. 
    Engage in natural conversation with users about whatever they ask.
    When appropriate, you can call the available tools to offer additional help.
    If you encounter a question or request that you cannot handle or answer confidently, use the escalate_to_representative tool to connect the user with a live representative.
    Never write code or output programming syntax - only call the provided tools using proper function calls.""",
    tools=[
        help_option_one,
        help_option_two,
        calculate_math_expression,
        escalate_to_representative,
    ],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,  # Balanced creativity and consistency
        top_p=0.95,       # Allow diverse but focused responses
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            ),
        ]
    ),
)
