from stable_baselines3 import PPO
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging
from config import load_config, get_project_root, load_prompt
from env import make_vec_env, make_eval_env
from rewards import GroundTruthRewardModel, ImplicitRewardModel, RewardModel
from llm_client import LLMClient, create_provider
from callbacks import SuccessRateCallback
from environments import get_adapter
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

load_dotenv()

def create_reward_model(adapter, config: dict, run_id: str, log_dir: Path) -> RewardModel:
    """Factory for reward models"""
    reward_type = config["reward_model"]
    
    if reward_type == "ground_truth":
        return GroundTruthRewardModel()
    
    elif reward_type == "implicit":
        provider = create_provider(config["llm_provider"])
        llm_client = LLMClient(provider, log_dir, run_id)
        return ImplicitRewardModel(
            llm_client=llm_client,
            env_id=config["env_string"],
            task_prompt=load_prompt(config["prompt_version"]),
            adapter=adapter,
        )
    
    else:
        raise ValueError(f"Unknown reward model: {reward_type}")
    
def main():
    config = load_config()
    adapter = get_adapter(config["env_string"])
    project_root = get_project_root()
    seed = config["seed"]
    
    # Build a run directory
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['reward_model']}_seed{seed}_{run_id}"
    run_dir = project_root / "experiments" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config snapshot for reproducibility
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Setup
    reward_model = create_reward_model(adapter, config, run_id, run_dir)
    logging.info(f"Reward model: {reward_model}")
    env = make_vec_env(adapter, config["env_string"], reward_model, seed=seed)
    eval_env = make_eval_env(adapter, config["env_string"], seed=seed+1000)
    
    callback = SuccessRateCallback(
        eval_env=eval_env,
        eval_freq=2000,
        n_eval_episodes=25,
        success_threshold=0.90,
        csv_path=run_dir / "metrics.csv",
        adapter=adapter,
    )
    
    # Train
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=str(run_dir / "tensorboard"), seed=seed)
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)
    model.save(run_dir / "model")
    
    logging.info(f"Run complete: {run_dir}")
    

if __name__ == "__main__":
    main()






