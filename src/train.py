from stable_baselines3 import PPO
from pathlib import Path
from utils import make_vec_env, get_reward_model
from datetime import datetime
from demo import run_demo
from utils import load_config
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

load_dotenv()
open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")

config = load_config()
env_string = config["env_string"]
total_timesteps = config["total_timesteps"]
reward_model_type = config["reward_model"]
reward_model = get_reward_model(reward_model_type)
env = make_vec_env(env_string, reward_model)

project_root = Path(__file__).parent.parent
models_dir = project_root / "models"
run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = models_dir / f"ppo_model_{run_id}"
tb_logs_path = project_root / "tensorboard-logs"

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=tb_logs_path)
model.learn(total_timesteps)
model.save(model_path)

run_demo(model_path, env_string, reward_model)





