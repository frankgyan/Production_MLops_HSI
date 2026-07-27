# check_config.py
import yaml

with open('config/production.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("✅ Config loaded successfully")
print(f"Stage 3 config: {config['stages']['stage3_train'].keys()}")
print(f"Model types: {config['stages']['stage3_train']['model_types']}")