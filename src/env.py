import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
from rewards import RewardModelWrapper, RewardModel
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.monitor import Monitor

def make_env(env_id: str, reward_model: RewardModel, seed: int = None, render_mode=None) -> gym.Env:
    """Training env with reward model wrapper."""
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    env = RewardModelWrapper(env, reward_model)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

def make_demo_env(env_id: str, render_mode="human") -> gym.Env:
    """Demo env for visualization — no reward model needed."""
    env = gym.make(env_id, render_mode=render_mode)
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

def make_vec_env(env_id: str, reward_model: RewardModel, seed: int = None, render_mode=None) -> VecEnv:
    env = DummyVecEnv([lambda: make_env(env_id, reward_model, seed, render_mode)])
    env = VecTransposeImage(env)
    return env

def make_eval_env(env_id, seed: int = None) -> VecEnv:
    """Env for running evals (without reward wrapper)"""
    def _make():
        env = gym.make(env_id, render_mode=None)
        env = RGBImgObsWrapper(env)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env

    env = DummyVecEnv([_make])
    env = VecTransposeImage(env)
    return env