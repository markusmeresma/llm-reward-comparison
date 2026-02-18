import gymnasium as gym
from rewards import RewardModelWrapper, RewardModel
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecEnv
from stable_baselines3.common.monitor import Monitor
from environments.adapter import EnvAdapter

def make_env(adapter: EnvAdapter, env_id: str, reward_model: RewardModel, seed: int = None, render_mode=None) -> gym.Env:
    env = adapter.make_base_env(env_id, seed, render_mode)
    env = RewardModelWrapper(env, reward_model, adapter)
    env = Monitor(env)
    if seed is not None:
        env.reset(seed=seed)
    return env

def make_vec_env(adapter: EnvAdapter, env_id: str, reward_model: RewardModel, seed: int = None, render_mode=None) -> VecEnv:
    env = DummyVecEnv([lambda: make_env(adapter, env_id, reward_model, seed, render_mode)])
    env = VecTransposeImage(env)
    return env

def make_demo_env(adapter: EnvAdapter, env_id: str, render_mode="human") -> gym.Env:
    """Demo env for visualization — no reward model needed."""
    env = adapter.make_base_env(env_id, render_mode=render_mode)
    return env

def make_eval_env(adapter: EnvAdapter, env_id, seed: int = None) -> VecEnv:
    """Env for running evals (without reward wrapper)"""
    def _make():
        env = adapter.make_base_env(env_id, seed=seed)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env

    env = DummyVecEnv([_make])
    env = VecTransposeImage(env)
    return env