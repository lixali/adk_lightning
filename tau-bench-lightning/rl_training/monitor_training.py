#!/usr/bin/env python3
"""Monitor training progress and GPU utilization"""

import time
import psutil
import GPUtil
import json
import os
from datetime import datetime

def monitor_resources():
    """Monitor system resources during training"""
    
    while True:
        timestamp = datetime.now().isoformat()
        
        # GPU utilization
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil * 100,
                'temperature': gpu.temperature
            })
        
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Training metrics (if available)
        metrics = {}
        if os.path.exists('rl_training/outputs/latest_metrics.json'):
            try:
                with open('rl_training/outputs/latest_metrics.json', 'r') as f:
                    metrics = json.load(f)
            except:
                pass
        
        # Print status
        print(f"\n{'='*60}")
        print(f"Time: {timestamp}")
        print(f"CPU: {cpu_percent:.1f}%")
        print(f"RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        print(f"\nGPU Status:")
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']} ({gpu['name']}): {gpu['load']:.1f}% load, "
                  f"{gpu['memory_util']:.1f}% memory ({gpu['memory_used']}MB/{gpu['memory_total']}MB), "
                  f"{gpu['temperature']}°C")
        
        if metrics:
            print(f"\nTraining Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_resources()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
