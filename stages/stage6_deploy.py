# stages/stage6_deploy.py
"""
STAGE 6: Model Deployment (Local)
NO EMOJIS - Windows compatible
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

def run_stage6(config_path):
    config = load_config(config_path)
    
    print("="*60)
    print("STAGE 6: Model Deployment")
    print("="*60)
    
    # Load model info
    registry_dir = Path(config['stages']['stage5_register']['artifact_dir'])
    
    if not (registry_dir / 'registry_info.json').exists():
        print("[ERROR] No registered model found. Run Stage 5 first.")
        return False
    
    with open(registry_dir / 'registry_info.json', 'r') as f:
        registry_info = json.load(f)
    
    print(f"\n[INFO] Deploying model: {registry_info['model_name']}")
    
    # Handle both MLflow and local registry formats
    if 'run_id' in registry_info:
        print(f"   Version: {registry_info['run_id'][:8]}")
    elif 'version' in registry_info:
        print(f"   Version: {registry_info['version']}")
    else:
        print(f"   Version: unknown")
    
    print(f"   Accuracy: {registry_info['test_accuracy']:.4f}")
    
    # Copy model to deployment location
    model_dir = Path(config['stages']['stage3_train']['artifact_dir'])
    deploy_dir = Path(config['stages']['stage6_deploy']['artifact_dir'])
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the model file
    src_model = model_dir / 'best_model.keras'
    
    # Also check if there's a model in registry
    if not src_model.exists():
        # Try to find model in registry
        registry_model = registry_dir / 'latest_model.keras'
        if registry_model.exists():
            src_model = registry_model
        else:
            # Try to find any .keras file in registry
            keras_files = list(registry_dir.glob('*.keras'))
            if keras_files:
                src_model = keras_files[0]
            else:
                print(f"[ERROR] No model file found.")
                return False
    
    # Copy to deployment
    shutil.copy(src_model, deploy_dir / 'deployed_model.keras')
    
    # Save deployment info
    deployment_info = {
        "model_name": registry_info['model_name'],
        "version": registry_info.get('version', registry_info.get('run_id', 'unknown')),
        "test_accuracy": registry_info['test_accuracy'],
        "deployment_target": config['stages']['stage6_deploy']['deployment_target'],
        "deployed_at": datetime.now().isoformat()
    }
    
    with open(deploy_dir / 'deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"\n[SUCCESS] Model deployed to: {deploy_dir}")
    print(f"   Model file: deployed_model.keras")
    print(f"   Deployment info saved")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    args = parser.parse_args()
    run_stage6(args.config)