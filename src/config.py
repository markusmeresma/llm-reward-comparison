from pathlib import Path
from typing import Any
import argparse
import yaml

ALL_SUPPORTED_MODELS = [
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5.2",
    "mistral-large-2512",
]

PROVIDER_BY_MODEL = {
    "openai/gpt-5-nano": "openrouter",
    "openai/gpt-5-mini": "openrouter",
    "openai/gpt-5.2": "openrouter",
    "mistral-large-2512": "mistral",
}


def infer_provider_for_model(model_name: str) -> str:
    provider = PROVIDER_BY_MODEL.get(model_name)
    if not provider:
        allowed = ", ".join(ALL_SUPPORTED_MODELS)
        raise ValueError(f"Unsupported --llm-model '{model_name}'. Allowed: {allowed}")
    return provider


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
    parser.add_argument(
        "--llm-model",
        required=False,
        choices=ALL_SUPPORTED_MODELS,
        help=(
            "Required for implicit reward model. "
            f"Supported: {', '.join(ALL_SUPPORTED_MODELS)}"
        ),
    )
    return parser.parse_args(argv)

def load_train_config(argv=None) -> dict[str, Any]:
    raw = load_config()
    args = parse_train_args(argv)
    
    env_key = args.env
    env_cfg = raw["envs"][env_key]
    defaults = raw["defaults"]
    
    provider = None
    
    # Enforce llm-model only for implicit runs
    if args.reward_model == "implicit":
        if not args.llm_model:
            raise ValueError("--llm-model is required when --reward-model implicit")
        if args.llm_model not in ALL_SUPPORTED_MODELS:
            allowed = ", ".join(ALL_SUPPORTED_MODELS)
            raise ValueError(f"Unsupported --llm-model '{args.llm_model}'. Allowed: {allowed}")
        provider = infer_provider_for_model(args.llm_model)

    
    resolved = {
        "env_alias": env_key,
        "env_string": env_cfg["env_string"],
        "reward_model": args.reward_model,  # ground_truth or implicit
        "total_timesteps": env_cfg["total_timesteps"],
        "eval_freq": env_cfg["eval_freq"],
        "n_eval_episodes": env_cfg["n_eval_episodes"],
        "prompt_version": env_cfg["prompt_version"],
        "llm_temperature": defaults["llm_temperature"],
        "seed": defaults["seed"],
        "segment_length": env_cfg["segment_length"],
    }
    
    if args.reward_model == "implicit":
        resolved["llm_provider"] = provider
        resolved["llm_model"] = args.llm_model
    
    # Only minigrid should carry this key
    if "success_threshold" in env_cfg:
        resolved["success_threshold"] = env_cfg["success_threshold"]
    
    return resolved