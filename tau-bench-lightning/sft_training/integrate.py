"""
Integration module to add SFT model provider to tau-bench.

This module modifies the tau-bench model provider system to support SFT fine-tuned models.
It adds the 'sft' provider to the existing provider system.

Usage:
1. Import this module in run.py or model provider files
2. Use --model-provider sft with appropriate model names
"""

import sys
import os
from pathlib import Path

# Add sft_training to path
sft_training_path = Path(__file__).parent
if str(sft_training_path) not in sys.path:
    sys.path.insert(0, str(sft_training_path))

def integrate_sft_provider():
    """
    Integrate SFT model provider into tau-bench.
    
    This function modifies the model provider system to include SFT models.
    Call this before running tau-bench with SFT models.
    """
    try:
        # Import tau-bench model providers
        from tau_bench.model_providers import get_model_provider
        from tau_bench.model_providers.base import ModelProvider
        
        # Import our SFT provider
        from sft_model_provider import SFTModelProvider, get_sft_model_provider
        
        # Register SFT provider
        # This is a simplified integration - you might need to modify based on
        # the actual tau-bench model provider architecture
        
        # If tau-bench uses a registry pattern, add SFT provider
        if hasattr(get_model_provider, '_providers'):
            get_model_provider._providers['sft'] = get_sft_model_provider
        
        print("SFT model provider integrated successfully")
        return True
        
    except ImportError as e:
        print(f"Error integrating SFT provider: {e}")
        print("Make sure tau-bench and required dependencies are installed")
        return False

def patch_run_script():
    """
    Apply patches to run.py to support SFT models.
    
    This function shows how to modify the main run script to support SFT models.
    """
    run_script_path = Path(__file__).parent.parent / "run.py"
    
    if not run_script_path.exists():
        print(f"run.py not found at {run_script_path}")
        return False
    
    # Read current run.py
    with open(run_script_path, 'r') as f:
        content = f.read()
    
    # Check if already patched
    if "sft_training.integrate" in content:
        print("run.py already patched for SFT support")
        return True
    
    # Find import section and add SFT integration
    import_section = "import argparse"
    if import_section in content:
        patch = f"""{import_section}

# SFT Model Integration
try:
    from sft_training.integrate import integrate_sft_provider
    integrate_sft_provider()
except ImportError:
    print("SFT model provider not available")"""
        
        content = content.replace(import_section, patch)
        
        # Write back
        with open(run_script_path, 'w') as f:
            f.write(content)
        
        print("run.py patched successfully for SFT support")
        return True
    else:
        print("Could not find appropriate location to patch run.py")
        return False

if __name__ == "__main__":
    print("Integrating SFT model provider...")
    success = integrate_sft_provider()
    
    if success:
        print("\nSFT provider integration complete!")
        print("\nYou can now use SFT models with:")
        print("python run.py --model-provider sft --model sft-gemma3-12b ...")
    else:
        print("\nIntegration failed. Please check dependencies.")
