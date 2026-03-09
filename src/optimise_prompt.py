"""CLI script for iterative prompt optimisation.

Usage (from project root):
    python src/optimise_prompt.py --env crafter --reward-type implicit \
      --experiment experiments/crafter_implicit_openai-gpt-5.2_seed42_... \
      --current-prompt prompts/crafter/implicit_v1.txt
"""
import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv
from config import load_config, get_project_root
from prompt_optimisation import optimise_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one iteration of prompt optimisation."
    )
    parser.add_argument(
        "--env", required=True, choices=["crafter"],
    )
    parser.add_argument(
        "--reward-type", required=True, choices=["implicit", "explicit"],
        help="Reward type whose prompt to optimise",
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Path to experiment directory containing crafter_eval_metrics.csv",
    )
    parser.add_argument(
        "--current-prompt", required=True,
        help="Path to the prompt file used in the experiment",
    )
    return parser.parse_args()

def main():
    load_dotenv()
    args = parse_args()
    
    raw = load_config()
    temperature = raw["defaults"]["optimiser_temperature"]
    
    project_root = get_project_root()
    history_path = (
        project_root / "optimised_prompts"
        / f"{args.env}_{args.reward_type}" / "history.yaml"
    )
    
    output_dir = optimise_prompt(
        env_alias=args.env,
        reward_type=args.reward_type,
        experiment_path=Path(args.experiment),
        current_prompt_path=Path(args.current_prompt),
        history_path=history_path,
        temperature=temperature,
    )
    
    print(f"\nOptimised prompt saved to:\n  {output_dir}")
    
if __name__ == "__main__":
    main()
    