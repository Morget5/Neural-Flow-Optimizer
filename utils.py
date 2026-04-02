import os
import json

def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)

def save_results(results, output_path):
    """
    Save optimization results to a JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def preprocess_data(data):
    """
    Preprocess input data for inference.
    """
    # Placeholder for actual preprocessing logic
    return data / 255.0
