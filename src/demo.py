from env import make_demo_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import PPO
from config import load_config
from environments import get_adapter
import argparse

# How to run from root folder: python3 src/demo.py models/<model_id> --env minigrid
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
def parse_demo_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a trained PPO model in demo mode.")
    parser.add_argument("model_path", help="Path to trained model")
    parser.add_argument(
        "--env",
        required=True,
        choices=["minigrid", "crafter"],
        help="Environment alias used in config.yaml under envs/",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_demo_args()
    config = load_config()
    env_id = config["envs"][args.env]["env_string"]
    adapter = get_adapter(env_id)
    run_demo(adapter, args.model_path, env_id)