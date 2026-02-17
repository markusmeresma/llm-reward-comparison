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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

load_dotenv()

def create_reward_model(config: dict, run_id: str, log_dir: Path) -> RewardModel:
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
        )
    
    else:
        raise ValueError(f"Unknown reward model: {reward_type}")
    
def main():
    config = load_config()
    project_root = get_project_root()
    
    # Training run identifiers and log paths
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    models_dir = project_root / "models"
    log_dir = project_root / "logs"
    # Running tensorboard locally: tensorboard --logdir tensorboard-logs
    tb_logs_path = project_root / "tensorboard-logs"
    model_path = models_dir / f"ppo_model_{run_id}"
    
    # Setup
    reward_model = create_reward_model(config, run_id, log_dir)
    logging.info(f"Reward model: {reward_model}")
    env = make_vec_env(config["env_string"], reward_model)
    eval_env = make_eval_env(config["env_string"])
    
    callback = SuccessRateCallback(
        eval_env=eval_env,
        eval_freq=2000,
        n_eval_episodes=25,
        success_threshold=0.90
    )
    
    # Train
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tb_logs_path)
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)
    model.save(model_path)
    
    logging.info(f"Model saved to {model_path}")
    

if __name__ == "__main__":
    main()






