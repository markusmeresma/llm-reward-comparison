"""CLI script for generating explicit reward functions via LLM.

Usage (from project root):
    python src/generate_reward.py --env crafter --llm-model openai/gpt-5.2
"""

import argparse
import logging

from pathlib import Path

from dotenv import load_dotenv
from config import load_config, load_prompt, ALL_SUPPORTED_MODELS
from explicit_generation import generate_reward_code

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reward generation.

    --env and --llm-model are required.
    """
    parser = argparse.ArgumentParser(
        description="Generate an explicit reward function via LLM."
    )
    parser.add_argument(
        "--env",
        required=True,
        choices=["minigrid", "crafter"],
        help="Environment alias",
    )
    parser.add_argument(
        "--llm-model",
        required=True,
        choices=ALL_SUPPORTED_MODELS,
        help=f"LLM model to use. Supported: {', '.join(ALL_SUPPORTED_MODELS)}",
    )
    parser.add_argument(
        "--prompt-path",
        required=False,
        help="Path to a prompt file (overrides explicit_prompt_version from config)",
    )
    return parser.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    
    raw = load_config()
    env_cfg = raw["envs"][args.env]
    defaults = raw["defaults"]

    prompt_version = env_cfg["explicit_prompt_version"]
    temperature = defaults["llm_temperature"]
    
    if args.prompt_path:
        prompt_text = Path(args.prompt_path).read_text().strip()
        prompt_source = args.prompt_path
    else:
        prompt_text = load_prompt(args.env, prompt_version)
        prompt_source = prompt_version
    
    output_dir = generate_reward_code(
        env_alias=args.env,
        llm_model=args.llm_model,
        prompt_text=prompt_text,
        prompt_source=prompt_source,
        temperature=temperature,
    )

    print(f"\nGenerated reward function saved to:\n  {output_dir}")
    
if __name__ == "__main__":
    main()
