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

import ast
import math
import operator
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
    """Calculate a mathematical expression and return the result using safe evaluation.
    
    This function safely evaluates mathematical expressions using Python's ast module
    and math functions. It supports basic arithmetic operations (+, -, *, /, //, %, **) 
    and common math functions (sin, cos, tan, log, sqrt, etc.).
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "1245/3.3", "sqrt(16)", "sin(0)")
        tool_context: The tool context for this invocation.
    
    Returns:
        The calculated result as a string.
        
    Examples:
        - "2 + 3 * 4" returns "The result of 2 + 3 * 4 is 14"
        - "sqrt(16)" returns "The result of sqrt(16) is 4.0"
        - "sin(math.pi/2)" returns "The result of sin(math.pi/2) is 1.0"
    """
    
    # Supported operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Supported math functions and constants
    safe_names = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        # Math module functions
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'atan2': math.atan2,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'asinh': math.asinh,
        'acosh': math.acosh,
        'atanh': math.atanh,
        'sqrt': math.sqrt,
        'log': math.log,
        'log10': math.log10,
        'log2': math.log2,
        'exp': math.exp,
        'ceil': math.ceil,
        'floor': math.floor,
        'fabs': math.fabs,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'degrees': math.degrees,
        'radians': math.radians,
        # Math constants
        'pi': math.pi,
        'e': math.e,
        'tau': math.tau,
        'inf': math.inf,
        'nan': math.nan,
        # Allow math module access
        'math': math,
    }
    
    def evaluate_node(node):
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.Name):
            if node.id in safe_names:
                return safe_names[node.id]
            else:
                raise ValueError(f"Unsupported name: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = evaluate_node(node.left)
            right = evaluate_node(node.right)
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = evaluate_node(node.operand)
            op = operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func = evaluate_node(node.func)
            args = [evaluate_node(arg) for arg in node.args]
            if callable(func):
                return func(*args)
            else:
                raise ValueError(f"Not a callable function: {func}")
        elif isinstance(node, ast.Attribute):
            # Handle attribute access like math.pi
            value = evaluate_node(node.value)
            attr_name = node.attr
            if hasattr(value, attr_name):
                return getattr(value, attr_name)
            else:
                raise ValueError(f"Attribute {attr_name} not found")
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
    
    try:
        # Parse the expression into an AST
        tree = ast.parse(expression, mode='eval')
        
        # Evaluate the expression safely
        result = evaluate_node(tree.body)
        
        # Format the result nicely
        if isinstance(result, float):
            if result.is_integer():
                result = int(result)
            else:
                result = round(result, 10)  # Limit decimal places
        
        return f"The result of {expression} is {result}"
        
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return f"Error calculating '{expression}': {str(e)}"
    except Exception as e:
        return f"Unexpected error calculating '{expression}': {str(e)}"


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
    instruction="""You are a helpful AI assistant with mathematical calculation capabilities. 
    Answer questions clearly and concisely. 
    Engage in natural conversation with users about whatever they ask.
    
    CRITICAL MATH DETECTION RULES:
    You MUST call calculate_math_expression when you detect ANY of these patterns:
    - Numbers with operators: "2+3", "10-4", "5*6", "15/3", "2^3" or "2**3"
    - Math keywords: "calculate", "solve", "what is", "compute", "find the result"
    - Math functions: "square root", "sqrt", "sin", "cos", "log", "factorial"
    - Percentage calculations: "15% of 200", "what percent is 25 of 100"
    - Word problems with numbers: "divide 1245 by 3.3", "multiply 7 times 8"
    - Any expression with parentheses: "(2+3)*4", "sqrt(16)", "sin(pi/2)"
    
    EXAMPLES OF WHEN TO CALL calculate_math_expression:
    - "What is 1245 divided by 3.3?" → call with "1245/3.3"
    - "Calculate 2 plus 3 times 4" → call with "2 + 3 * 4"
    - "Square root of 16" → call with "sqrt(16)"
    - "What's 15% of 200?" → call with "200 * 0.15"
    - "Sin of pi over 2" → call with "sin(pi/2)"
    
    NEVER attempt to calculate math yourself - ALWAYS use the tool.
    
    If you encounter a question or request that you cannot handle or answer confidently, 
    use the escalate_to_representative tool to connect the user with a live representative.
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
