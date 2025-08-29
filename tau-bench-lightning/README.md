# tau-bench: Gemini Integration

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


### 2. Model Factory Integration  
- ✅ Added Google platform case to `model_factory()` in `tau_bench/model_utils/model/general_model.py`
- ✅ Supports both API key and service account authentication

### 3. Provider Support
- ✅ Uses `vertex_ai` provider (works with default Google Cloud credentials)
- ✅ Supports `gemini` provider (requires GOOGLE_API_KEY)
- ✅ Tool calling functionality verified and working

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

#### Option 2: Service Account (Production)
```bash
# Set service account credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"

# Or authenticate with service account
gcloud auth activate-service-account --key-file=/path/to/your/service-account-key.json
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

### Why No Separate License?
- **Gemini models are Google Cloud services** accessed through Vertex AI
- **Your existing Google Cloud authentication** provides access
- **Usage is pay-per-use** and billed to your Google Cloud project
- **No separate API keys needed** when using `vertex_ai` provider

## Test Results:
- ✅ Basic completion: Working
- ✅ Tool calling: Working  
- ✅ User simulation: Working
- ✅ Full tau-bench run: Working
- ✅ Results generation: Working (saved to `results/` folder)

## Models Supported:
- `gemini-2.5-flash` (recommended)

## Usage Examples:

### Basic Command (your request):
```bash
python run.py --agent-strategy tool-calling --env retail --model gemini-2.5-flash --model-provider vertex_ai --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1
```

### With Gemini-1.5-flash Agent + Gemini-2.5-flash User:
```bash
python run.py --agent-strategy tool-calling --env retail --model gemini-1.5-flash --model-provider vertex_ai --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 1
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
python analyze_results.py results/tool-calling-gemini-2.5-flash-0.0_range_0--1_user-gemini-2.5-flash-llm_0826161338.json
```

### Sample Output:
```
📊 Analyzing results from: results/tool-calling-gemini-2.5-flash-0.0_range_0--1_user-gemini-2.5-flash-llm_0826161338.json
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
python analyze_results.py results/tool-calling-gemini-1.5-flash-0.0_range_0--3_user-gemini-2.5-flash-llm_TIMESTAMP.json
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
  --user-model gemini-2.5-flash --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 3 --end-index 115
```



### Deployment Options

#### 1. HuggingFace (Hosted)
```bash
export HUGGINGFACE_API_KEY=your_hf_token_here
python run.py --agent-strategy tool-calling --env retail \
  --model gemma-2-9b-it --model-provider gemma \
  --user-model gemini-2.5-flash --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 1
```

#### 2. Ollama (Local)
```bash
# Install and pull models
ollama pull gemma:7b
ollama pull gemma3:12b

# Run benchmark with Gemma 7B
python run.py --agent-strategy tool-calling --env retail \
  --model gemma:7b --model-provider gemma \
  --user-model gemini-2.5-flash --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 3 --end-index 115

# Run benchmark with Gemma3 12B (latest, higher capability)
python run.py --agent-strategy tool-calling --env retail \
  --model gemma3:12b --model-provider gemma \
  --user-model gemini-2.5-flash --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 3 --end-index 115
```

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

#### 4. Custom Fine-tuned Model (Future)
```bash
# For your future fine-tuned models
python run.py --agent-strategy tool-calling --env retail \
  --model /path/to/your/fine-tuned-gemma --model-provider gemma --base-url http://your-server:8000/v1 \
  --user-model gemini-2.5-flash --user-model-provider vertex_ai \
  --user-strategy llm --max-concurrency 1
```

### Testing Available Models
Use the provided test script to check what's available in your environment:

```bash
python test_gemma_integration.py
```

This will test:
- HuggingFace API access
- Local Ollama models  
- Integration functionality
- Generate usage examples

### Implementation Features
- ✅ **Multi-provider support**: HuggingFace, Ollama, vLLM, custom endpoints
- ✅ **Future-ready**: Easy switching to fine-tuned models via model path/URL
- ✅ **Error handling**: Graceful fallbacks and clear error messages
- ✅ **Flexible deployment**: Works with local, hosted, and custom setups
- ✅ **Latest models**: Includes Gemma3 12B with enhanced performance
