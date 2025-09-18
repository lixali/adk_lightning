# vLLM Gemma3 Debug

This directory contains code to debug Gemma3 model calls using vLLM via LiteLLM.

## Setup

### 1. Start vLLM Server

First, you need to start a vLLM server to serve the Gemma3 model:

```bash
# Option 1: Use the provided script
./start_vllm_server.sh

# Option 2: Start manually
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-12b-it \
    --port 8000 \
    --host 0.0.0.0 \
    --served-model-name gemma3:12b \
    --trust-remote-code \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192
```

The server will be available at `http://localhost:8000` and provide an OpenAI-compatible API.

### 2. Run the Debug Script

Once the vLLM server is running, execute the debug script:

```bash
cd src/
python vllm_gemma3_prompt_debug.py
```

This will:
- Run task 0 by default
- Show detailed logging of what gets sent to Gemma3
- Save logs to timestamped files
- Display the exact JSON payload and raw input text

## Configuration

You can modify these parameters:

- `--task-id`: Which task to run (default: 0)
- `--model`: Model name (default: gemma3:12b)
- `--model-provider`: Provider (default: vllm)

Example:
```bash
python vllm_gemma3_prompt_debug.py --task-id 5 --model gemma3:12b
```

## Key Features

- **Raw Input Logging**: Shows exactly what text is fed to Gemma3
- **HTTP Request/Response**: Captures the complete API interaction
- **JSON Payload**: Displays the exact parameters sent to vLLM
- **Detailed Logs**: Saves comprehensive logs for later analysis

## Troubleshooting

### Error: "module 'torch._inductor' has no attribute 'config'"

This means LiteLLM is trying to use vLLM locally instead of via HTTP. Make sure:

1. Your vLLM server is running on `http://localhost:8000`
2. The `vllm_base_url` in the agent is correct
3. You're using `provider="vllm"` (which gets mapped to OpenAI-compatible API)

### Connection Refused

Make sure the vLLM server is running and accessible:

```bash
curl http://localhost:8000/v1/models
```

Should return a list of available models.

```
vllm-gemma3-debug
├── src
│   ├── vllm_gemma3_prompt_debug.py       # Entry point for using the vllm provider with the gemma3 agent model.
│   ├── agents
│   │   └── vllm_tool_calling_agent.py     # Defines the VLLMToolCallingAgent class for vllm-specific functionality.
│   ├── utils
│   │   └── __init__.py                     # Initialization file for the utils package.
│   └── __init__.py                         # Initialization file for the src package.
├── requirements.txt                        # Lists the dependencies required for the project.
├── setup.py                                # Setup script for the project.
└── README.md                                # Documentation for the project.
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd vllm-gemma3-debug
pip install -r requirements.txt
```

## Usage

To run the vllm gemma3 debug environment, execute the following command:

```bash
python src/vllm_gemma3_prompt_debug.py
```

This will initialize the environment and agent, set up the configuration, and run the tasks using the vllm provider.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.