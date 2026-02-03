from pathlib import Path
from typing import Any
import yaml

def load_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
    
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_prompt(name: str) -> str:
    prompt_path = get_project_root() / "prompts" / f"{name}.txt"
    return prompt_path.read_text().strip()