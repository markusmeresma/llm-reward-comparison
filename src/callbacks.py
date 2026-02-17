import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import csv
from pathlib import Path
import time

class SuccessRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, success_threshold, csv_path: Path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_threshold = success_threshold
        self.timestep_to_threshold = None
        self.csv_path = csv_path
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "success_rate", "mean_return", "mean_episode_length", "wall_time_s"])
        
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        successes = 0
        episode_returns = []
        episode_lengths = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_return = 0.0
            ep_length = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                done = done[0]
                ep_return += float(reward[0])
                ep_length += 1
            
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            if float(reward[0]) > 0:
                successes += 1
                
        success_rate = successes / self.n_eval_episodes
        mean_return = np.mean(episode_returns)
        wall_time = time.time() - self.start_time
        
        # Log to TensorBoard
        self.logger.record("eval/success_rate", success_rate)
        self.logger.record("eval/mean_return", mean_return)
        self.logger.dump(self.num_timesteps)
        
        # Log to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.num_timesteps, success_rate, round(mean_return, 4), round(np.mean(episode_lengths), 1), round(wall_time, 1)])
        
        if success_rate >= self.success_threshold:
            self.timestep_to_threshold = self.num_timesteps
            print(f"Threshold reached at {self.num_timesteps} timesteps (success_rate={success_rate:.2f})")
            return False
        
        return True