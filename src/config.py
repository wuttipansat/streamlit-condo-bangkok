from pathlib import Path
import yaml
from utils import get_project_root

def load_config(config_path: str = "config.yaml") -> dict:

    with open(Path(get_project_root() / config_path), "r") as file:
        return yaml.safe_load(file)
    

