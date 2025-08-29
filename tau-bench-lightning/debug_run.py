#!/usr/bin/env python3
"""
Debug version of run.py with verbose output
"""

import sys
sys.path.insert(0, '/home/lixali_google_com/work/tau-bench')

from tau_bench.types import RunConfig
from tau_bench.run import run

def main():
    print("🐛 Debug run with Gemini")
    
    # Create config
    config = RunConfig(
        num_trials=1,
        env="retail",
        model="gemini-2.5-flash",
        model_provider="vertex_ai",
        user_model="gemini-2.5-flash",
        user_model_provider="vertex_ai",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        start_index=0,
        end_index=1,
        task_ids=None,
        log_dir="results",
        max_concurrency=1,
        seed=42,
        shuffle=False,
        user_strategy="llm",
        few_shot_displays_path=None
    )
    
    print(f"Config: {config}")
    print("Starting run...")
    
    try:
        results = run(config)
        print(f"✓ Run completed! Results: {len(results)} items")
        for result in results:
            print(f"  - Task {result.task_id}: reward={result.reward}")
    except Exception as e:
        print(f"❌ Run failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
