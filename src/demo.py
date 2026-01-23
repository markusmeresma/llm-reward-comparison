from utils import make_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from utils import load_config
from utils import get_reward_model
import sys

def run_demo(model_path, env_string, reward_model):
    demo_env = DummyVecEnv([lambda: make_env(env_string, reward_model, render_mode="human")])
    demo_env = VecTransposeImage(demo_env)
    obs = demo_env.reset()
    model = PPO.load(model_path)
    print(f"Running demo for {model_path}")
    
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, _, dones, _ = demo_env.step(action)
        if dones.any():
            obs = demo_env.reset()
            
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python demo.py <model_path>")
        sys.exit(1)
    
    config = load_config()
    env_string = config["env_string"]
    reward_model = get_reward_model(config["reward_model"])
    model_path = sys.argv[1]
    run_demo(model_path, env_string, reward_model)