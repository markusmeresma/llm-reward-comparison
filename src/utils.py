import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from rewards import GroundTruthRewardModel, RewardModelWrapper, ImplicitRewardModel
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from pathlib import Path
import yaml
from rewards import RewardModel
from typing import Any
from stable_baselines3.common.vec_env import VecEnv

def load_config() -> dict[str, Any]:
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def make_env(env_string, reward_model, render_mode=None) -> gym.Env:
    env = gym.make(env_string, render_mode=render_mode)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = RewardModelWrapper(env, reward_model)
    return env

def make_vec_env(env_string, reward_model, render_mode=None) -> VecEnv:
    env = DummyVecEnv([lambda: make_env(env_string, reward_model, render_mode)])
    env = VecTransposeImage(env)
    return env

def get_reward_model(reward_model_type) -> RewardModel:
    if reward_model_type == "ground_truth":
        return GroundTruthRewardModel()
    elif reward_model_type == "implicit":
        return ImplicitRewardModel()
    else:
        raise ValueError(f"Invalid reward model type: {reward_model_type}")