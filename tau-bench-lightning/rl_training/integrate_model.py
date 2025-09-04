#!/usr/bin/env python3
"""
Model Integration Script
Integrates the trained Gemma3 model back into tau-bench for evaluation.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelIntegrator:
    """Integrates trained model back into tau-bench ecosystem"""
    
    def __init__(self, trained_model_path: str, integration_name: str = "trained-gemma3-12b"):
        self.trained_model_path = Path(trained_model_path)
        self.integration_name = integration_name
        self.tau_bench_root = Path(__file__).parent.parent
        
        if not self.trained_model_path.exists():
            raise ValueError(f"Trained model path does not exist: {trained_model_path}")
    
    def integrate_model(self):
        """Main integration process"""
        logger.info(f"🔧 Starting model integration for {self.integration_name}")
        
        # Step 1: Validate trained model
        self.validate_trained_model()
        
        # Step 2: Create vLLM deployment configuration
        self.create_vllm_config()
        
        # Step 3: Update tau-bench model configurations
        self.update_tau_bench_configs()
        
        # Step 4: Create evaluation scripts
        self.create_evaluation_scripts()
        
        # Step 5: Create comparison scripts
        self.create_comparison_scripts()
        
        logger.info("✅ Model integration completed successfully!")
        self.print_usage_instructions()
    
    def validate_trained_model(self):
        """Validate that the trained model has all necessary files"""
        logger.info("🔍 Validating trained model...")
        
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "training_config.json"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.trained_model_path / file).exists():
                missing_files.append(file)
        
        if missing_files:
            logger.warning(f"⚠️  Missing files: {missing_files}")
            logger.warning("Model may still work, but some files are missing")
        
        # Check for model weights
        weight_files = list(self.trained_model_path.glob("*.safetensors")) + \
                     list(self.trained_model_path.glob("*.bin"))
        
        if not weight_files:
            raise ValueError("No model weight files found (.safetensors or .bin)")
        
        logger.info(f"✅ Model validation completed. Found {len(weight_files)} weight files")
    
    def create_vllm_config(self):
        """Create vLLM deployment configuration"""
        logger.info("⚙️  Creating vLLM deployment configuration...")
        
        vllm_config = {
            "model_path": str(self.trained_model_path.absolute()),
            "model_name": self.integration_name,
            "tensor_parallel_size": 8,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "host": "0.0.0.0",
            "port": 8001,  # Different port to avoid conflicts
            "trust_remote_code": True
        }
        
        config_path = self.tau_bench_root / "rl_training" / "vllm_config.json"
        with open(config_path, 'w') as f:
            json.dump(vllm_config, f, indent=2)
        
        # Create deployment script
        deploy_script = f"""#!/bin/bash
# Deploy trained Gemma3 model with vLLM

echo "🚀 Deploying trained Gemma3 model: {self.integration_name}"

# Check if model path exists
if [ ! -d "{self.trained_model_path}" ]; then
    echo "❌ Model path not found: {self.trained_model_path}"
    exit 1
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \\
    --model {self.trained_model_path} \\
    --host 0.0.0.0 \\
    --port 8001 \\
    --tensor-parallel-size 8 \\
    --gpu-memory-utilization 0.9 \\
    --max-model-len 4096 \\
    --served-model-name "{self.integration_name}" \\
    --trust-remote-code \\
    --enable-chunked-prefill \\
    --max-num-batched-tokens 8192

echo "✅ Trained model deployed at http://localhost:8001"
echo "Model name: {self.integration_name}"
"""
        
        deploy_script_path = self.tau_bench_root / "rl_training" / "deploy_trained_model.sh"
        with open(deploy_script_path, 'w') as f:
            f.write(deploy_script)
        
        os.chmod(deploy_script_path, 0o755)
        logger.info(f"✅ vLLM deployment script created: {deploy_script_path}")
    
    def update_tau_bench_configs(self):
        """Update tau-bench configuration to support the trained model"""
        logger.info("🔧 Updating tau-bench configurations...")
        
        # Update the gemma.py model file to include our trained model
        gemma_model_file = self.tau_bench_root / "tau_bench" / "model_utils" / "model" / "gemma.py"
        
        if gemma_model_file.exists():
            # Read current content
            with open(gemma_model_file, 'r') as f:
                content = f.read()
            
            # Add our trained model to the pricing map (if not already there)
            trained_model_config = f'''
# Trained model configuration
PRICE_PER_INPUT_TOKEN_MAP["{self.integration_name}"] = 0.3 / 1000000
PRICE_PER_OUTPUT_TOKEN_MAP["{self.integration_name}"] = 0.9 / 1000000
CAPABILITY_SCORE_MAP["{self.integration_name}"] = 0.95  # Improved through RL training
LATENCY_MS_PER_OUTPUT_TOKEN_MAP["{self.integration_name}"] = 30.0
'''
            
            if self.integration_name not in content:
                # Add the configuration before the last line
                lines = content.split('\n')
                insert_point = -5  # Insert before the end of the file
                lines.insert(insert_point, trained_model_config)
                
                with open(gemma_model_file, 'w') as f:
                    f.write('\n'.join(lines))
                
                logger.info("✅ Updated gemma.py with trained model configuration")
        
        # Update environment model mappings
        user_env_file = self.tau_bench_root / "tau_bench" / "envs" / "user.py"
        if user_env_file.exists():
            with open(user_env_file, 'r') as f:
                content = f.read()
            
            # Add vLLM provider mapping for our trained model
            vllm_mapping = f'''        elif "{self.integration_name}" in model:
            return "vllm"'''
            
            if self.integration_name not in content:
                # Find the gemma provider section and add our mapping
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'elif "vllm/" in model:' in line:
                        lines.insert(i, vllm_mapping)
                        break
                
                with open(user_env_file, 'w') as f:
                    f.write('\n'.join(lines))
                
                logger.info("✅ Updated user.py with vLLM provider mapping")
    
    def create_evaluation_scripts(self):
        """Create scripts to evaluate the trained model"""
        logger.info("📊 Creating evaluation scripts...")
        
        # Create evaluation script for trained model
        eval_script = f"""#!/bin/bash
# Evaluate trained Gemma3 model on tau-bench

echo "📊 Evaluating trained model: {self.integration_name}"

# Make sure the model server is running
if ! curl -s http://localhost:8001/health > /dev/null; then
    echo "❌ Model server not running. Please start it first:"
    echo "   ./rl_training/deploy_trained_model.sh"
    exit 1
fi

echo "✅ Model server is running"

# Run evaluation on retail environment
echo "🛍️  Evaluating on retail environment..."
python run.py \\
    --agent-strategy tool-calling \\
    --env retail \\
    --model {self.integration_name} \\
    --model-provider vllm \\
    --user-model gemini-2.5-flash \\
    --user-model-provider vertex_ai \\
    --user-strategy llm \\
    --max-concurrency 4 \\
    --end-index 50 \\
    --seed 42

# Run evaluation on airline environment  
echo "✈️  Evaluating on airline environment..."
python run.py \\
    --agent-strategy tool-calling \\
    --env airline \\
    --model {self.integration_name} \\
    --model-provider vllm \\
    --user-model gemini-2.5-flash \\
    --user-model-provider vertex_ai \\
    --user-strategy llm \\
    --max-concurrency 4 \\
    --end-index 50 \\
    --seed 42

echo "✅ Evaluation completed!"
echo "📁 Check results/ directory for output files"
"""
        
        eval_script_path = self.tau_bench_root / "rl_training" / "evaluate_trained_model.sh"
        with open(eval_script_path, 'w') as f:
            f.write(eval_script)
        
        os.chmod(eval_script_path, 0o755)
        
        # Create error analysis script for trained model
        error_analysis_script = f"""#!/bin/bash
# Analyze errors from trained model evaluation

echo "🔍 Analyzing errors from trained model evaluation..."

# Find the latest results file for the trained model
LATEST_RETAIL=$(ls -t results/*{self.integration_name}*retail*.json 2>/dev/null | head -1)
LATEST_AIRLINE=$(ls -t results/*{self.integration_name}*airline*.json 2>/dev/null | head -1)

if [ -n "$LATEST_RETAIL" ]; then
    echo "📊 Analyzing retail results: $LATEST_RETAIL"
    python llm_error_analysis.py "$LATEST_RETAIL"
fi

if [ -n "$LATEST_AIRLINE" ]; then
    echo "📊 Analyzing airline results: $LATEST_AIRLINE"  
    python llm_error_analysis.py "$LATEST_AIRLINE"
fi

echo "✅ Error analysis completed!"
"""
        
        error_analysis_path = self.tau_bench_root / "rl_training" / "analyze_trained_model_errors.sh"
        with open(error_analysis_path, 'w') as f:
            f.write(error_analysis_script)
        
        os.chmod(error_analysis_path, 0o755)
        
        logger.info("✅ Evaluation scripts created")
    
    def create_comparison_scripts(self):
        """Create scripts to compare trained model with baseline"""
        logger.info("📈 Creating comparison scripts...")
        
        comparison_script = f"""#!/usr/bin/env python3
\"\"\"
Compare trained model performance with baseline Gemma3
\"\"\"

import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_results(pattern):
    \"\"\"Load results from JSON files matching pattern\"\"\"
    files = glob.glob(pattern)
    results = []
    
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            results.extend(data)
    
    return results

def calculate_metrics(results):
    \"\"\"Calculate performance metrics\"\"\"
    total_tasks = len(results)
    successful_tasks = sum(1 for task in results if task.get('reward', 0) == 1.0)
    success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
    
    return {{
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'failed_tasks': total_tasks - successful_tasks,
        'success_rate': success_rate
    }}

def compare_models():
    \"\"\"Compare baseline vs trained model performance\"\"\"
    
    # Load baseline results (original Gemma3)
    baseline_pattern = "results/tool-calling-gemma3:12b-*.json"
    baseline_results = load_results(baseline_pattern)
    
    # Load trained model results
    trained_pattern = "results/*{self.integration_name}*.json"
    trained_results = load_results(trained_pattern)
    
    if not baseline_results:
        print("❌ No baseline results found. Run baseline evaluation first.")
        return
    
    if not trained_results:
        print("❌ No trained model results found. Run trained model evaluation first.")
        return
    
    # Calculate metrics
    baseline_metrics = calculate_metrics(baseline_results)
    trained_metrics = calculate_metrics(trained_results)
    
    # Print comparison
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("=" * 50)
    print(f"Baseline Model (gemma3:12b):")
    print(f"  Success Rate: {{baseline_metrics['success_rate']:.2%}}")
    print(f"  Successful: {{baseline_metrics['successful_tasks']}}/{{baseline_metrics['total_tasks']}}")
    
    print(f"\\nTrained Model ({self.integration_name}):")
    print(f"  Success Rate: {{trained_metrics['success_rate']:.2%}}")
    print(f"  Successful: {{trained_metrics['successful_tasks']}}/{{trained_metrics['total_tasks']}}")
    
    # Calculate improvement
    improvement = trained_metrics['success_rate'] - baseline_metrics['success_rate']
    print(f"\\n📈 Improvement: {{improvement:+.2%}}")
    
    if improvement > 0:
        print("✅ Trained model shows improvement!")
    elif improvement < 0:
        print("❌ Trained model shows regression.")
    else:
        print("➖ No significant change.")
    
    # Save comparison report
    report = {{
        'baseline': baseline_metrics,
        'trained': trained_metrics,
        'improvement': improvement,
        'comparison_date': pd.Timestamp.now().isoformat()
    }}
    
    with open('rl_training/comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\\n💾 Detailed report saved to: rl_training/comparison_report.json")

if __name__ == "__main__":
    compare_models()
"""
        
        comparison_script_path = self.tau_bench_root / "rl_training" / "compare_models.py"
        with open(comparison_script_path, 'w') as f:
            f.write(comparison_script)
        
        os.chmod(comparison_script_path, 0o755)
        
        # Create complete evaluation pipeline
        pipeline_script = f"""#!/bin/bash
# Complete evaluation pipeline for trained model

echo "🚀 Starting complete evaluation pipeline for {self.integration_name}"

# Step 1: Deploy the model
echo "📡 Step 1: Deploying trained model..."
./rl_training/deploy_trained_model.sh &
MODEL_PID=$!

# Wait for model to be ready
echo "⏳ Waiting for model to be ready..."
sleep 30

# Check if model is responding
for i in {{1..10}}; do
    if curl -s http://localhost:8001/health > /dev/null; then
        echo "✅ Model is ready!"
        break
    fi
    echo "⏳ Waiting... ($i/10)"
    sleep 10
done

# Step 2: Run evaluations
echo "📊 Step 2: Running evaluations..."
./rl_training/evaluate_trained_model.sh

# Step 3: Analyze errors
echo "🔍 Step 3: Analyzing errors..."
./rl_training/analyze_trained_model_errors.sh

# Step 4: Compare with baseline
echo "📈 Step 4: Comparing with baseline..."
python rl_training/compare_models.py

# Step 5: Clean up
echo "🧹 Step 5: Cleaning up..."
kill $MODEL_PID 2>/dev/null || true

echo "✅ Complete evaluation pipeline finished!"
echo "📁 Check rl_training/ directory for all reports and analyses"
"""
        
        pipeline_path = self.tau_bench_root / "rl_training" / "run_complete_evaluation.sh"
        with open(pipeline_path, 'w') as f:
            f.write(pipeline_script)
        
        os.chmod(pipeline_path, 0o755)
        
        logger.info("✅ Comparison and pipeline scripts created")
    
    def print_usage_instructions(self):
        """Print usage instructions"""
        print("\\n" + "="*60)
        print("🎉 MODEL INTEGRATION COMPLETED!")
        print("="*60)
        print(f"\\n🤖 Trained Model: {self.integration_name}")
        print(f"📁 Model Path: {self.trained_model_path}")
        print("\\n📋 USAGE INSTRUCTIONS:")
        print("\\n1️⃣ Deploy the trained model:")
        print("   ./rl_training/deploy_trained_model.sh")
        print("\\n2️⃣ Evaluate the trained model:")
        print("   ./rl_training/evaluate_trained_model.sh")
        print("\\n3️⃣ Analyze errors:")
        print("   ./rl_training/analyze_trained_model_errors.sh")
        print("\\n4️⃣ Compare with baseline:")
        print("   python rl_training/compare_models.py")
        print("\\n5️⃣ Run complete pipeline:")
        print("   ./rl_training/run_complete_evaluation.sh")
        print("\\n🔗 TAU-BENCH INTEGRATION:")
        print(f"   Use model name: {self.integration_name}")
        print("   Use provider: vllm")
        print("   Example command:")
        print(f"   python run.py --model {self.integration_name} --model-provider vllm ...")
        print("\\n📊 MONITORING:")
        print("   - Check results/ directory for evaluation outputs")
        print("   - Check rl_training/ directory for analysis reports")
        print("   - Monitor GPU usage during evaluation")
        print("\\n" + "="*60)

def main():
    """Main integration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate trained Gemma3 model into tau-bench")
    parser.add_argument("model_path", help="Path to the trained model directory")
    parser.add_argument("--name", default="trained-gemma3-12b", help="Integration name for the model")
    
    args = parser.parse_args()
    
    try:
        integrator = ModelIntegrator(args.model_path, args.name)
        integrator.integrate_model()
    except Exception as e:
        logger.error(f"❌ Integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
