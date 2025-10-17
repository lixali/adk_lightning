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
      <a href="https://github.com/google/adk-web">ADK Web</a>.
    </h3>
</html>

This implementation now includes native **Gemma 3 12B model using ollma**. 

---

## ✨ Key Features

- **Native Gemma 3 Model Integration**: Support for Gemma 3 12B model with 
  intelligent caching, automatic model loading, and seamless ADK integration.


## 🔥 Gemma 3 Model Integration

This implementation includes native support for **Google's Gemma 3 12B model** with several key optimizations:

## 🚀 Follwoing is the instruction to run. 


1. Install additional dependencies:

```bash
pip install google-adk transformers torch accelerate ollama
```

2. Download gemma3-12b using ollama:

```bash

ollama pull gemma3:12b
```

3. Start ollama server:

```bash

ollama serve

```


## 🏁 Feature Highlight

### Define a single agent with Gemma 3 with system prompt and function calls from Tau Bench:

```python

root_agent = Agent(
    model=LiteLlm(model="ollama_chat/gemma3:12b"),
    name="retail_support_agent",
    description=(
        "Helpful customer service agent in retail domain with access to order management, user lookup, and product information tools."
    ),
    instruction=retail_wiki,
    tools=[
        calculate,
        cancel_pending_order,
        exchange_delivered_order_items,
        find_user_id_by_email,
        find_user_id_by_name_zip,
        get_order_details,
        get_product_details,
        get_user_details,
        list_all_product_types,
        modify_pending_order_address,
        modify_pending_order_items,
        modify_pending_order_payment,
        modify_user_address,
        return_delivered_order_items,
        think,
        transfer_to_human_agents,
    ],
)

```



###  Evaluate Agents

```bash
adk web --host 0.0.0.0 contributing/samples

```

And then open browser and type in http://0.0.0.0:8000/dev-ui/?app=hello_world_ollama (or http://localhost:38577/dev-ui/?app=hello_world_ollama ; either one of the links should work) to interact with the agent




## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
