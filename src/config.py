from pathlib import Path
from typing import Any
import argparse
import yaml

def load_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
    
def get_project_root() -> Path:
    return Path(__file__).parent.parent

def load_prompt(env_alias: str, name: str) -> str:
    prompt_path = get_project_root() / "prompts" / env_alias / f"{name}.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found for env='{env_alias}', version='{name}': {prompt_path}"
        )

    content = prompt_path.read_text().strip()
    if not content:
        raise ValueError(
            f"Prompt file is empty for env='{env_alias}', version='{name}': {prompt_path}"
        )

    return content

def parse_train_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO with selected env and reward model.")
    parser.add_argument(
        "--env",
        required=True,
        choices=["minigrid", "crafter"],
        help="Environment alias: minigrid -> MiniGrid-Empty-5x5-v0, crafter -> CrafterReward-v1",
    )
    parser.add_argument(
        "--reward-model",
        required=True,
        choices=["ground_truth", "implicit"],
        help="Reward model alias: ground_truth -> ground_truth, implicit -> implicit"
    )
    return parser.parse_args(argv)

def load_train_config(argv=None) -> dict[str, Any]:
    raw = load_config()
    args = parse_train_args(argv)
    
    env_key = args.env
    env_cfg = raw["envs"][env_key]
    defaults = raw["defaults"]
    
    resolved = {
        "env_alias": env_key,
        "env_string": env_cfg["env_string"],
        "reward_model": args.reward_model,  # ground_truth or implicit
        "total_timesteps": env_cfg["total_timesteps"],
        "eval_freq": env_cfg["eval_freq"],
        "n_eval_episodes": env_cfg["n_eval_episodes"],
        "prompt_version": env_cfg["prompt_version"],
        "llm_provider": defaults["llm_provider"],
        "seed": defaults["seed"],
        "segment_length": env_cfg["segment_length"],
    }
    
    # Only minigrid should carry this key
    if "success_threshold" in env_cfg:
        resolved["success_threshold"] = env_cfg["success_threshold"]
    
    return resolved