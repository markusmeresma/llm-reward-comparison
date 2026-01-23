import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path
from utils import make_vec_env, get_reward_model
from datetime import datetime
from demo import run_demo

env_string = "MiniGrid-SimpleCrossingS11N5-v0"
total_timesteps = 50_000
reward_model_type = "ground_truth"
reward_model = get_reward_model(reward_model_type)
env = make_vec_env(env_string, reward_model)

model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps)

project_root = Path(__file__).parent.parent
models_dir = project_root / "models"
model_path = models_dir / f"ppo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
model.save(model_path)

run_demo(model_path, env_string, reward_model)





