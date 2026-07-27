# run_full_pipeline.py
"""
Full MLOps Pipeline Orchestrator
Runs all stages in order with error handling
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_stage(script_name, config_path):
    """Run a stage script."""
    cmd = ["python", f"stages/{script_name}", "--config", config_path]
    print(f"\n[RUNNING] {' '.join(cmd)}")
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        # Filter out common warnings that don't affect execution
        stderr_lines = result.stderr.split('\n')
        for line in stderr_lines:
            if 'cudart64' not in line and 'nvcuda' not in line and 'cuda_driver' not in line:
                if line.strip():
                    print(f"[WARNING] {line}")
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/production.yaml')
    parser.add_argument('--start_from', default='stage1',
                       choices=['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'])
    args = parser.parse_args()
    
    stages = [
        ('stage1_ingest.py', 'Stage 1: Data Ingestion'),
        ('stage2_preprocess.py', 'Stage 2: Preprocessing'),
        ('stage3_train_simple.py', 'Stage 3: Model Training'),
        ('stage4_evaluate.py', 'Stage 4: Evaluation'),
        ('stage5_register_local.py', 'Stage 5: Model Registry'),  # Local version
        ('stage6_deploy.py', 'Stage 6: Deployment'),
    ]
    
    # Find start index
    start_idx = 0
    for i, (script, name) in enumerate(stages):
        if args.start_from in script:
            start_idx = i
            break
    
    print("="*60)
    print("MLOps Pipeline Execution")
    print("="*60)
    print(f"Starting from: {args.start_from}")
    print(f"Config: {args.config}")
    
    for script, name in stages[start_idx:]:
        print(f"\n{'='*60}")
        print(f"[STAGE] {name}")
        print(f"{'='*60}")
        
        if not run_stage(script, args.config):
            print(f"[FAILED] {name} FAILED. Pipeline halted.")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("[SUCCESS] FULL PIPELINE COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()


# # run_full_pipeline.py
# """
# Full MLOps Pipeline Orchestrator
# Runs all stages in order with error handling
# NO EMOJIS - Windows compatible
# """

# import subprocess
# import sys
# import argparse
# from pathlib import Path

# def run_stage(script_name, config_path):
#     """Run a stage script."""
#     cmd = ["python", f"stages/{script_name}", "--config", config_path]
#     print(f"\n[RUNNING] {' '.join(cmd)}")
    
#     # Use stdout/stderr directly (Python 3.6 compatible)
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    
#     # Print output
#     if result.stdout:
#         print(result.stdout)
#     if result.stderr:
#         print(f"[ERRORS]\n{result.stderr}")
    
#     return result.returncode == 0

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', default='config/production.yaml')
#     parser.add_argument('--start_from', default='stage1',
#                        choices=['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6'])
#     args = parser.parse_args()
    
#     stages = [
#         ('stage1_ingest.py', 'Stage 1: Data Ingestion'),
#         ('stage2_preprocess.py', 'Stage 2: Preprocessing'),
#         ('stage3_train_simple.py', 'Stage 3: Model Training'),
#         ('stage4_evaluate.py', 'Stage 4: Evaluation'),
#         ('stage5_register.py', 'Stage 5: Model Registry'),
#         ('stage6_deploy.py', 'Stage 6: Deployment'),
#     ]
    
#     # Find start index
#     start_idx = 0
#     for i, (script, name) in enumerate(stages):
#         if args.start_from in script:
#             start_idx = i
#             break
    
#     print("="*60)
#     print("MLOps Pipeline Execution")
#     print("="*60)
#     print(f"Starting from: {args.start_from}")
#     print(f"Config: {args.config}")
    
#     for script, name in stages[start_idx:]:
#         print(f"\n{'='*60}")
#         print(f"[STAGE] {name}")
#         print(f"{'='*60}")
        
#         if not run_stage(script, args.config):
#             print(f"[FAILED] {name} FAILED. Pipeline halted.")
#             sys.exit(1)
    
#     print("\n" + "="*60)
#     print("[SUCCESS] FULL PIPELINE COMPLETE!")
#     print("="*60)

# if __name__ == "__main__":
#     main()