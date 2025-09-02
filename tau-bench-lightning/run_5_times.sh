#!/bin/bash

# Script to run the tau-bench experiment 5 times sequentially
# Each run will be executed one after another (not in parallel)

echo "Starting 5 sequential runs of tau-bench experiment..."
echo "Command: python run.py --agent-strategy tool-calling --env airline --model gemma3:12b --model-provider gemma --user-model gemini-2.5-flash --user-model-provider vertex_ai --user-strategy llm --max-concurrency 3 --end-index 50"
echo "=========================================="

for i in {1..8}; do
    echo ""
    echo "Run $i/5 - Starting at $(date)"
    echo "----------------------------------------"
    
    python run.py --agent-strategy tool-calling --env airline \
      --model gemma3:12b --model-provider gemma \
      --user-model gemini-2.5-flash --user-model-provider vertex_ai \
      --user-strategy llm --max-concurrency 6 --end-index 50
    
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "Run $i/5 - Completed successfully at $(date)"
    else
        echo "Run $i/5 - Failed with exit code $exit_code at $(date)"
        echo "Stopping execution due to error."
        exit $exit_code
    fi
    
    echo "----------------------------------------"
    
    # Optional: Add a small delay between runs (uncomment if needed)
    # echo "Waiting 10 seconds before next run..."
    # sleep 10
done

echo ""
echo "All 5 runs completed successfully!"
echo "Finished at $(date)"
echo "=========================================="
