from env import make_demo_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from config import load_config
from environments import get_adapter
import sys

# How to run from root folder: python3 src/demo.py models/<model_id>
def run_demo(adapter, model_path, env_id):
    demo_env = DummyVecEnv([lambda: make_demo_env(adapter, env_id)])
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
    env_id = config["env_string"]
    adapter = get_adapter(config["env_string"])
    model_path = sys.argv[1]
    run_demo(adapter, model_path, env_id)