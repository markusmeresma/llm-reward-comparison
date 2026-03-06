from stable_baselines3 import PPO
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import logging
from config import get_project_root, load_prompt, load_train_config
from env import make_vec_env, make_eval_env
from rewards import GroundTruthRewardModel, ImplicitRewardModel, ExplicitRewardModel, RewardModel
from llm_client import LLMClient, create_provider
from callbacks import MiniGridCallback, CrafterCallback
from segment_rollout_buffer import SegmentRolloutBuffer
from environments import get_adapter
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

load_dotenv()

def create_reward_model(adapter, config: dict, run_id: str, log_dir: Path) -> RewardModel:
    """Factory for reward models. Selects between ground truth (environment
    native rewards) and implicit (segment-based LLM evaluation) based on
    the --reward-model CLI flag."""
    reward_type = config["reward_model"]
    
    if reward_type == "ground_truth":
        return GroundTruthRewardModel()
    
    elif reward_type == "implicit":
        provider = create_provider(
            config["llm_provider"], config["llm_model"], config["llm_temperature"]
        )
        llm_client = LLMClient(provider, log_dir, run_id)
        return ImplicitRewardModel(
            llm_client=llm_client,
            task_prompt=load_prompt(config["env_alias"], config["prompt_version"]),
            adapter=adapter,
            segment_length=config["segment_length"]
        )
    
    elif reward_type =="explicit":
        return ExplicitRewardModel(reward_code_path=config["reward_code"])
    
    else:
        raise ValueError(f"Unknown reward model: {reward_type}")
    
def create_callback(adapter, config: dict, eval_env, run_dir: Path, reward_model: RewardModel):
    """Factory for environment-specific training callbacks.
    Passes the reward model to CrafterCallback so it can read episode scores
    in implicit mode (where step-level rewards are always 0.0)."""
    env_id = config["env_string"]
    
    if env_id.startswith("MiniGrid"):
        return MiniGridCallback(
            adapter=adapter,
            eval_env=eval_env,
            eval_freq=config["eval_freq"],
            n_eval_episodes=config["n_eval_episodes"],
            success_threshold=config["success_threshold"],
            csv_path=run_dir / "metrics.csv",
        )
        
    if env_id.startswith("Crafter"):
        return CrafterCallback(
            adapter=adapter,
            eval_env=eval_env,
            eval_freq=config["eval_freq"],
            n_eval_episodes=config["n_eval_episodes"],
            train_episode_csv_path=run_dir / "crafter_train_episodes.csv",
            train_achievements_csv_path=run_dir / "crafter_train_achievements.csv",
            eval_csv_path=run_dir / "crafter_eval_metrics.csv",
            reward_model=reward_model,
        )
        
    raise ValueError(f"No callback configured for environment: {env_id}")
    
def main():
    config = load_train_config()
    adapter = get_adapter(config["env_string"])
    project_root = get_project_root()
    seed = config["seed"]
    
    # Build a run directory
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{config['env_alias']}_{config['reward_model']}_seed{seed}_{run_id}"
    if config["reward_model"] == "implicit":
        # Extract the model tag to identify which model this training run used
        model_tag = config["llm_model"].replace("/", "-")
        run_name = f"{config['env_alias']}_{config['reward_model']}_{model_tag}_seed{seed}_{run_id}"
    elif config["reward_model"] == "explicit":
        # Extract the generation directory name (e.g. "crafter_openai-gpt-5.2_20260303_120000")
        # to identify which generated code this training run used
        reward_dir_name = Path(config["reward_code"]).parent.name
        run_name = f"{config['env_alias']}_{config['reward_model']}_{reward_dir_name}_seed{seed}_{run_id}"
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
    
    callback = create_callback(
        adapter=adapter,
        config=config,
        eval_env=eval_env,
        run_dir=run_dir,
        reward_model=reward_model
    )
    
    ppo_kwargs = dict(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=seed,
    )
    
    # When using implicit rewards, PPO needs a custom rollout buffer that
    # writes segment scores into step rewards before GAE computation.
    # min_segment_length=16 floors the per-step divisor to prevent extreme
    # reward amplification from very short partial segments.
    if isinstance(reward_model, ImplicitRewardModel):
        ppo_kwargs["rollout_buffer_class"] = SegmentRolloutBuffer
        ppo_kwargs["rollout_buffer_kwargs"] = {
            "reward_model": reward_model,
            "min_segment_length": 16,
        }
        
    model = PPO(**ppo_kwargs)
    
    # Segment length must divide n_steps so that segment boundaries align
    # with rollout boundaries (no partial segments from misalignment).
    if isinstance(reward_model, ImplicitRewardModel):
        segment_length = config["segment_length"]
        assert model.n_steps % segment_length == 0, (
            f"n_steps ({model.n_steps}) must be divisible by segment_length ({segment_length})"
        )
    
    model.learn(total_timesteps=config["total_timesteps"], callback=callback)
    model.save(run_dir / "model")
    logging.info(f"Run complete: {run_dir}")
    

if __name__ == "__main__":
    main()






