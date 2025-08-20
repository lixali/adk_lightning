# Agent Development Kit (ADK)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/google-adk)](https://pypi.org/project/google-adk/)
[![Python Unit Tests](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml/badge.svg)](https://github.com/google/adk-python/actions/workflows/python-unit-tests.yml)
[![r/agentdevelopmentkit](https://img.shields.io/badge/Reddit-r%2Fagentdevelopmentkit-FF4500?style=flat&logo=reddit&logoColor=white)](https://www.reddit.com/r/agentdevelopmentkit/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/google/adk-python)

<html>
    <h2 align="center">
      <img src="https://raw.githubusercontent.com/google/adk-python/main/assets/agent-development-kit.png" width="256"/>
    </h2>
    <h3 align="center">
      An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
    </h3>
    <h3 align="center">
      Important Links:
      <a href="https://google.github.io/adk-docs/">Docs</a>,
      <a href="https://github.com/google/adk-samples">Samples</a>,
      <a href="https://github.com/google/adk-java">Java ADK</a> &
      <a href="https://github.com/google/adk-web">ADK Web</a>.
    </h3>
</html>

This implementation now includes native **Gemma 3 12B model support** with model caching and efficient memory management. ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.


---

## ✨ Key Features

- **Native Gemma 3 Model Integration**: Support for Gemma 3 12B model with 
  intelligent caching, automatic model loading, and seamless ADK integration.


## 🤖 Agent2Agent (A2A) Protocol and ADK Integration

For remote agent-to-agent communication, ADK integrates with the
[A2A protocol](https://github.com/google-a2a/A2A/).
See this [example](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/agents)
for how they can work together.

## 🔥 Gemma 3 Model Integration

This implementation includes native support for **Google's Gemma 3 12B model** with several key optimizations:

## 🚀 Installation

### Stable Release (Recommended)

You can install the latest stable version of ADK using `pip`:

```bash
pip install google-adk
```

### Additional Requirements for Gemma 3 Model

To use the Gemma 3 12B model, you'll need:

1. A HuggingFace account and token with access to the Gemma 3 model
2. Set your HuggingFace token as an environment variable:

```bash
export HUGGING_FACE_TOKEN="your_huggingface_token_here"
```

3. Install additional dependencies:

```bash
pip install transformers torch accelerate
```


## 📚 Documentation

Explore the full documentation for detailed guides on building, evaluating, and
deploying agents:

* **[Documentation](https://google.github.io/adk-docs)**

## 🏁 Feature Highlight

### Define a single agent with Gemma 3:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemma-12b", # Native Gemma 3 12B model support
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a single agent with Gemini:

```python
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

### Define a multi-agent system:

Define a multi-agent system with coordinator agent, greeter agent, and task execution agent. You can mix Gemma 3 and Gemini models as needed. Then ADK engine and the model will guide the agents works together to accomplish the task.

```python
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents - mix of Gemma 3 and Gemini models
greeter = LlmAgent(name="greeter", model="gemma-12b", ...)
task_executor = LlmAgent(name="task_executor", model="gemini-2.0-flash", ...)

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemma-12b",  # Using Gemma 3 12B for coordination
    description="I coordinate greetings and tasks using Gemma 3.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_executor
    ]
)
```

### Development UI

A built-in development UI to help you test, evaluate, debug, and showcase your agent(s). Now supports both Gemma 3 and Gemini models.

<img src="https://raw.githubusercontent.com/google/adk-python/main/assets/adk-web-dev-ui-function-call.png"/>

### Run your Gemma 3 agent:

```bash
# Set your HuggingFace token first
export HUGGING_FACE_TOKEN="your_token_here"

# Run the hello world sample with Gemma 3
python contributing/samples/hello_world/main.py
```

###  Evaluate Agents

```bash
adk web --host 0.0.0.0 contributing/samples

```

And then open browser and type in http://0.0.0.0:8000/dev-ui/?app=hello_world to interact with the agent


## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
