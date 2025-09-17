# tau-bench: Gemma3-12B and Gemini Integration

Following commands needs to be run to install necessary packages 

```

pip install litellm
pip install google-generativeai

gcloud auth login
```

To use Gemini-model vertex AI , install the following (this is not required to install to run this framework because litellm is using ollama to call Gemini)

```

pip install google-cloud-aiplatform

```

## ✅ Integration Complete!

The Gemini-2.5-flash and Gemma3 model has been successfully integrated into the tau-bench codebase. You can now run the exact command you requested:

```bash
python run.py --agent-strategy tool-calling --env retail --model gemini-2.5-flash --model-provider vertex_ai --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1
```

## What was implemented:

### 1. Model Platform Support
- ✅ Added `GOOGLE = "google"` to the `Platform` enum in `tau_bench/model_utils/model/model.py`
- ✅ Created `GoogleModel` class in `tau_bench/model_utils/model/google.py` with full feature support:
  - Chat completion via litellm

## 🏗️ Model Architecture: No Model "Instantiation" - Just API Calls

### Key Insight: No Models are Actually Loaded in Python
- **Agent Model**: Not an instantiated model - just configuration strings
- **User Model**: Not an instantiated model - just configuration strings  
- **Both models**: Use the same LiteLLM → Ollama → Gemma3:12b pipeline
- **Only difference**: Different temperatures and tool configurations

### Agent Model Calling Location
**File**: `tau_bench/agents/tool_calling_agent.py` (Lines 101-107)
```python
# THE AGENT MODEL API CALL - Line 101
res = completion(
    messages=messages,                    # Conversation history
    model=self.litellm_model,            # "gemma3:12b" (STRING)
    custom_llm_provider=self.litellm_provider,  # "ollama" (STRING)
    tools=self.tools_info,               # 16 retail tools (ARRAY)
    temperature=self.temperature,         # 0.0 (FLOAT - deterministic)
)
```

### User Model Calling Location  
**File**: `tau_bench/envs/user.py` (Lines 63-70)
```python
# THE USER MODEL API CALL - Line 65
res = completion(
    model=self.model,                    # "gemma3:12b" (STRING)
    custom_llm_provider=mapped_provider, # "ollama" (STRING)
    messages=messages,                   # User conversation history
    temperature=self.temperature         # 1.0 (FLOAT - creative)
)
```

### Model Configuration Setup
**Agent Model Setup** (`tau_bench/agents/tool_calling_agent.py:45-48`):
```python
if provider == "gemma" and ":" in model:
    self.litellm_provider = "ollama"     # STRING
    self.litellm_model = "gemma3:12b"    # STRING
```

**User Model Setup** (`tau_bench/envs/user.py:15-17`):
```python
if provider == "gemma" and "gemma3:12b" in model:
    return "ollama"                      # STRING returned
```

### Execution Flow Summary
1. **Command Line** → Two separate configuration objects created
2. **Agent Config**: `ToolCallingAgent(model="gemma3:12b", provider="gemma", temperature=0.0)`
3. **User Config**: `LLMUserSimulationEnv(model="gemma3:12b", provider="gemma", temperature=1.0)`
4. **Runtime**: Both make separate HTTP calls to the same Ollama server
5. **Ollama**: Serves the same Gemma3:12b model weights to both API calls

| Component | Location | Purpose | Model Call |
|-----------|----------|---------|------------|
| **Agent Model** | `tool_calling_agent.py:101` | Task solving with tools | `completion()` with tools + temp=0.0 |
| **User Model** | `user.py:65` | Natural user responses | `completion()` with no tools + temp=1.0 |

## Prompt Locations in the Codebase

### Policy Input Prompts (for the trained policy)
The input prompts that define the policy behavior are located in:

**Retail Domain:**
- **File:** `tau_bench/envs/retail/wiki.md` (Lines 1-82)
- **Purpose:** Complete retail agent policy with authentication, order management, returns, exchanges
- **Key Sections:** User authentication, order cancellation, modifications, returns, exchanges
- **Usage:** Loaded via `tau_bench/envs/retail/wiki.py` and integrated into agent prompt in `tau_bench/agents/chat_react_agent.py`

**Airline Domain:**
- **File:** `tau_bench/envs/airline/wiki.md` (Lines 1-71) 
- **Purpose:** Airline agent policy for flight booking, modification, cancellation
- **Key Sections:** Flight booking, reservation modifications, baggage policies, payment rules
- **Usage:** Loaded via `tau_bench/envs/airline/wiki.py` and integrated into agent prompt

**Agent Instruction Templates:**
- **File:** `tau_bench/agents/chat_react_agent.py` (Lines 100-199)
- **Components:**
  - `REACT_INSTRUCTION`: Full reasoning format with "Thought:" and "Action:" steps
  - `ACT_INSTRUCTION`: Direct action format without explicit reasoning
- **Integration:** Combined with wiki policy content to form complete system prompt (Line 29-31)

### Reward Prompts (for training evaluation)
The reward model prompts are located in:

**Reward Model Implementation:**
- **File:** `rl_training/reward_model.py` (Lines 50-140)
- **Method:** `create_reward_prompt()` 
- **Content:** Detailed evaluation criteria with weighted scoring:
  - Decision Making Quality (40%)
  - Factual Accuracy (30%) 
  - Task Completion (20%)
  - User Experience (10%)
- **Examples:** Includes one-shot examples for "Good" vs "Bad" responses
- **Output:** Returns structured evaluation with score, reasoning, and criteria breakdown

**Integration Points:**
- Reward prompts are used during PPO training in `rl_training/ppo_trainer.py`
- Episode evaluation happens via `rl_training/tau_bench_data_collector.py`
- Gemini-2.5-pro model serves as the reward evaluator with detailed scoring rubric

## Authentication Setup

### How Authentication Works
When you use `--model-provider vertex_ai`, litellm automatically uses Google Cloud's **Application Default Credentials (ADC)**. No separate API keys or licenses are needed for Gemini models - they're accessed through your existing Google Cloud authentication.

### Verify Your Authentication
Check if you're properly authenticated:

```bash
# Check active Google Cloud accounts
gcloud auth list

# Check current project
gcloud config get-value project

# Verify Vertex AI API is enabled
gcloud services list --enabled --filter="name:aiplatform.googleapis.com"
```

### Authentication Options

#### Option 1: User Account
```bash
# Login with your Google account
gcloud auth login

# Set application default credentials
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### Troubleshooting Authentication

#### Check Your Setup
```bash
# Verify authentication status
gcloud auth list

# Test Vertex AI access (optional)
gcloud ai models list --region=us-central1
```

#### Common Issues
- **"Authentication failed"**: Run `gcloud auth application-default login`
- **"Project not found"**: Check `gcloud config get-value project`
- **"API not enabled"**: Run `gcloud services enable aiplatform.googleapis.com`
- **"Permission denied"**: Ensure your account has Vertex AI permissions

## Usage Examples:

### Basic Command:
```bash
python run.py --agent-strategy tool-calling --env retail --model gemini-2.5-pro --model-provider vertex_ai --user-model gemini-2.5-pro --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1
```

### With gemma3-12b Agent + Gemini-2.5-pro User:
```bash
python run.py --agent-strategy tool-calling --env retail --model gemma3:12b --model-provider gemma --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1
```

### With Different Models:
```bash
python run.py --agent-strategy tool-calling --env retail --model gemini-1.5-pro --model-provider vertex_ai --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1
```

### With API Key (if needed):
```bash
export GOOGLE_API_KEY="your-api-key"
python run.py --agent-strategy tool-calling --env retail --model gemini-2.5-flash --model-provider gemini --user-model gemini-2.5-flash --user-model-provider gemini --user-strategy llm --max-concurrency 1
```

## Files Created/Modified:
1. `tau_bench/model_utils/model/model.py` - Added Google platform
2. `tau_bench/model_utils/model/google.py` - New Google model implementation  
3. `tau_bench/model_utils/model/general_model.py` - Added Google to model factory
4. `test_gemini_setup.py` - Setup verification script
5. `test_providers.py` - Provider testing script

## Analyzing Results

After running benchmarks, you can analyze the results using the included analysis script:

### Basic Analysis Command:
```bash
python analyze_results.py results/your-results-file.json
```

### Example with Your Gemini Results:
```bash
python analyze_results.py results/tool-calling-gemini-2.5-flash-0.0_range_0--1_user-gemini-2.5-flash-llm_0826161338_concurrency-1.json
```

### Sample Output:
```
📊 Analyzing results from: results/tool-calling-gemini-2.5-flash-0.0_range_0--1_user-gemini-2.5-flash-llm_0826161338_concurrency-1.json
📝 Total entries: 115
🔄 Number of trials: 1
🏆 Average reward: 0.5043 (50.43% pass rate)
📋 Unique tasks: 115

📈 Pass@k Metrics:
  Pass@1: 0.5043 (50.43%)

📊 Additional Stats:
  Tasks with at least 1 success: 58/115 (50.43%)
```

### For Multi-Trial Analysis:
Run with multiple trials to see Pass@k for k > 1:
```bash
python run.py --agent-strategy tool-calling --env retail --model gemini-1.5-flash --model-provider vertex_ai --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1 --num-trials 3
```

Then analyze:
```bash
python analyze_results.py results/tool-calling-gemini-1.5-flash-0.0_range_0--3_user-gemini-2.5-flash-llm_concurrency-1_TIMESTAMP.json
```

This will show Pass@1, Pass@2, and Pass@3 metrics.

### Understanding the Metrics:
- **Average Reward**: Overall success rate (0.0-1.0, where 1.0 = 100% success)
- **Pass^k**: Success rate of all K attempts pass

The integration is now complete and working exactly as requested! 🎉

---

## Previous Documentation
For the original tau-bench documentation, see [README_old.md](README_old.md).

## 🆕 Gemma Model Support

### New Platform Added: GEMMA

We've added comprehensive support for Google's Gemma models as agent models, with flexibility for both hosted and custom fine-tuned deployments:

### Recommended Command (Gemma 12B + Gemini User):
```bash
python run.py --agent-strategy tool-calling --env retail \
  --model gemma3:12b --model-provider gemma \
  --user-model gemini-2.5-pro --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 3 --end-index 115
```

### Run a Specific Task (for debugging):
```bash
# Run just task 0 for debugging
python run.py --agent-strategy tool-calling --env retail \
  --model gemma3:12b --model-provider gemma \
  --user-model gemini-2.5-pro --user-model-provider vertex_ai \
  --user-strategy llm --task-id 0

# Run task 5 for debugging
python run.py --agent-strategy tool-calling --env retail \
  --model gemma3:12b --model-provider gemma \
  --user-model gemini-2.5-pro --user-model-provider vertex_ai \
  --user-strategy llm --task-id 5

# Alternative: Run multiple specific tasks
python run.py --agent-strategy tool-calling --env retail \
  --model gemma3:12b --model-provider gemma \
  --user-model gemini-2.5-pro --user-model-provider vertex_ai \
  --user-strategy llm --task-ids 0 5 10
```

## 🐛 Debugging Features

### Single Task Execution
The `--task-id` flag is perfect for debugging specific tasks:

```bash
# Debug task 0 with detailed output
python run.py --agent-strategy tool-calling --env retail \
  --model gemma3:12b --model-provider gemma \
  --user-model gemini-2.5-pro --user-model-provider vertex_ai \
  --user-strategy llm --task-id 0
```

### Other Useful Debugging Flags
```bash
# Run with single concurrency for cleaner output
--max-concurrency 1

# Run specific task range
--start-index 5 --end-index 10

# Run multiple specific tasks
--task-ids 0 2 5 8

# Add custom suffix to output files
--custom-suffix "_debug_run"
```

#### . Ollama (Local)
```bash
# Install and pull models
ollama pull gemma:7b
ollama pull gemma3:12b

#### 3. vLLM (Multi-GPU)
```bash
# Start vLLM server
vllm serve google/gemma-2-9b-it --host 0.0.0.0 --port 8000

# Run benchmark
python run.py --agent-strategy tool-calling --env retail \
  --model gemma-2-9b-it --model-provider gemma --base-url http://localhost:8000/v1 \
  --user-model gemini-2.5-flash --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 1
```