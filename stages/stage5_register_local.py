# stages/stage5_register_local.py
"""
STAGE 5: Simple Local Model Registry (No MLflow)
No server needed - just copies the model file
"""

import json
import yaml
import argparse
import shutil
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_stage5(config_path):
    config = load_config(config_path)
    
    print("="*60)
    print("STAGE 5: Model Registry (Local)")
    print("="*60)
    
    # Load evaluation results
    eval_dir = Path(config['stages']['stage4_evaluate']['artifact_dir'])
    
    if not (eval_dir / 'evaluation_results.json').exists():
        print("[ERROR] Evaluation results not found. Run Stage 4 first.")
        return False
    
    with open(eval_dir / 'evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    # Check if validation passed
    status_file = eval_dir / 'validation_status.txt'
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = f.read().strip()
        
        if "FAILED" in status:
            print("[WARNING] Validation failed. Skipping registration.")
            print(f"   Status: {status}")
            return False
    
    # Create registry directory
    registry_dir = Path(config['stages']['stage5_register']['artifact_dir'])
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model to registry
    model_dir = Path(config['stages']['stage3_train']['artifact_dir'])
    src_model = model_dir / 'best_model.keras'
    
    if not src_model.exists():
        print(f"[ERROR] Model not found: {src_model}")
        return False
    
    # Create versioned model name
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_model = registry_dir / f'model_v{version}.keras'
    shutil.copy(src_model, dst_model)
    
    # Also copy as latest
    shutil.copy(src_model, registry_dir / 'latest_model.keras')
    
    # Save registry info
    registry_info = {
        "model_name": config['stages']['stage5_register']['registered_model_name'],
        "version": version,
        "model_path": str(dst_model),
        "test_accuracy": eval_results['test_accuracy'],
        "test_loss": eval_results['test_loss'],
        "timestamp": datetime.now().isoformat()
    }
    
    with open(registry_dir / 'registry_info.json', 'w') as f:
        json.dump(registry_info, f, indent=2)
    
    print(f"[SUCCESS] Model registered: {registry_info['model_name']}")
    print(f"   Version: {registry_info['version']}")
    print(f"   Test Accuracy: {registry_info['test_accuracy']:.4f}")
    print(f"   Location: {registry_dir}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    args = parser.parse_args()
    run_stage5(args.config)