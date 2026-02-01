from utils import make_env, get_reward_model
from utils import load_config

config = load_config()
env_string = config["env_string"]
reward_model = get_reward_model(config["reward_model"])
env = make_env(env_string, reward_model)

obs, info = env.reset()

print("After reset:")
# Access trajectory through the wrapper chain
print(env.env.trajectory)

obs, reward, term, trunc, info = env.step(2)
print("\nAfter step (forward):")
print(env.env.trajectory)

obs, reward, term, trunc, info = env.step(1)
print("\nAfter step (turn right):")
print(env.env.trajectory)


