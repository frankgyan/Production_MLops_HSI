# stages/stage5_register.py
"""
STAGE 5: Model Registry with MLflow
NO EMOJIS - Windows compatible
"""

import json
import yaml
import argparse
import tensorflow as tf
from pathlib import Path
from datetime import datetime

try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARNING] MLflow not installed. Install with: pip install mlflow")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_stage5(config_path):
    config = load_config(config_path)
    
    print("="*60)
    print("STAGE 5: Model Registry")
    print("="*60)
    
    if not MLFLOW_AVAILABLE:
        print("[ERROR] MLflow not available. Skipping registry.")
        print("[INFO] Install with: pip install mlflow")
        return False
    
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
    
    # Load the model
    model_dir = Path(config['stages']['stage3_train']['artifact_dir'])
    model_path = model_dir / 'best_model.keras'
    
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return False
    
    # Load the model as a Keras object
    model = tf.keras.models.load_model(model_path)
    
    # Setup MLflow
    tracking_uri = config['stages']['stage5_register']['mlflow_tracking_uri']
    
    # If using localhost and server not running, fallback to local file
    if "localhost" in tracking_uri or "127.0.0.1" in tracking_uri:
        try:
            mlflow.set_tracking_uri(tracking_uri)
        except:
            print("[WARNING] MLflow server not running. Using local file tracking.")
            mlflow.set_tracking_uri("./mlflow_local")
    else:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment(config['stages']['stage5_register']['experiment_name'])
    
    with mlflow.start_run(run_name="model_registration"):
        # Log model - FIXED: Use the model object directly
        mlflow.tensorflow.log_model(
            model,  # The model object goes here as positional argument
            artifact_path="model",
            registered_model_name=config['stages']['stage5_register']['registered_model_name']
        )
        
        # Log metrics
        mlflow.log_metrics({
            "test_accuracy": eval_results['test_accuracy'],
            "test_loss": eval_results['test_loss']
        })
        
        # Log parameters
        mlflow.log_params({
            "model_type": "spectral_spatial",
            "test_samples": 561
        })
        
        # Save registry info
        registry_dir = Path(config['stages']['stage5_register']['artifact_dir'])
        registry_dir.mkdir(parents=True, exist_ok=True)
        
        registry_info = {
            "model_name": config['stages']['stage5_register']['registered_model_name'],
            "run_id": mlflow.active_run().info.run_id,
            "test_accuracy": eval_results['test_accuracy'],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(registry_dir / 'registry_info.json', 'w') as f:
            json.dump(registry_info, f, indent=2)
    
    print(f"[SUCCESS] Model registered: {registry_info['model_name']}")
    print(f"   Run ID: {registry_info['run_id']}")
    print(f"   Test Accuracy: {registry_info['test_accuracy']:.4f}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    args = parser.parse_args()
    run_stage5(args.config)
