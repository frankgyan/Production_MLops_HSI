# stages/stage4_evaluate.py
"""
STAGE 4: Model Evaluation on Test Set
NO EMOJIS - Windows compatible
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
import yaml
import argparse
from pathlib import Path
from datetime import datetime

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_stage4(config_path):
    config = load_config(config_path)
    
    print("="*60)
    print("STAGE 4: Model Evaluation")
    print("="*60)
    
    # Load test data
    preprocess_dir = Path(config['stages']['stage2_preprocess']['artifact_dir'])
    X_test = np.load(preprocess_dir / 'X_test.npy')
    y_test = np.load(preprocess_dir / 'y_test.npy')
    
    # Load model
    model_dir = Path(config['stages']['stage3_train']['artifact_dir'])
    model = tf.keras.models.load_model(model_dir / 'best_model.keras')
    
    print(f"\n[INFO] Evaluating on {len(X_test)} test samples...")
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, 4), verbose=0)
    
    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Metrics
    class_names = ['HNHP', 'HNLP', 'LNHP', 'LNLP']
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Results
    results = {
        "test_accuracy": float(accuracy),
        "test_loss": float(loss),
        "classification_report": report,
        "confusion_matrix": conf_matrix.tolist(),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    eval_dir = Path(config['stages']['stage4_evaluate']['artifact_dir'])
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    with open(eval_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[SUCCESS] Test Accuracy: {accuracy:.4f}")
    print(f"[INFO] Results saved to: {eval_dir}")
    
    # Check threshold
    threshold = config['stages']['stage4_evaluate']['min_accuracy_threshold']
    passed = accuracy >= threshold
    
    with open(eval_dir / 'validation_status.txt', 'w') as f:
        f.write("PASSED" if passed else f"FAILED (Accuracy: {accuracy:.4f} < {threshold})")
    
    print(f"\n[INFO] Validation {'PASSED' if passed else 'FAILED'} (Threshold: {threshold})")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    args = parser.parse_args()
    run_stage4(args.config)