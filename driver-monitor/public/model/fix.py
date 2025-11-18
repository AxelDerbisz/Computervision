import json
import os

# 1. Read the current model file
try:
    with open('model.json', 'r') as f:
        data = json.load(f)
        print("Loaded model.json")
except FileNotFoundError:
    print("Error: model.json not found in this folder.")
    exit()

# 2. Fix the InputLayer config
fixed = False
try:
    # Navigate to the layers config
    layers = data['modelTopology']['model_config']['config']['layers']
    
    for layer in layers:
        if layer['class_name'] == 'InputLayer':
            config = layer['config']
            # The core fix: Rename 'batch_shape' to 'batch_input_shape'
            if 'batch_shape' in config:
                config['batch_input_shape'] = config.pop('batch_shape')
                fixed = True
                print("Fixed: Renamed 'batch_shape' to 'batch_input_shape'")
            break
            
    if not fixed:
        print("Warning: 'batch_shape' not found. The file might already be fixed.")

except KeyError as e:
    print(f"Error parsing JSON structure: {e}")

# 3. Save the fixed file
if fixed:
    with open('model.json', 'w') as f:
        json.dump(data, f)
    print("Success! model.json has been updated.")
else:
    print("No changes made.")