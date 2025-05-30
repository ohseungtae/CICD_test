import yaml
import os

def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
