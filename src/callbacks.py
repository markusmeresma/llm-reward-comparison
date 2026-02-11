import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback

class SuccessRateCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes, success_threshold):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_threshold = success_threshold
        self.timestep_to_threshold = None
        
    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                done = done[0]
                
            if float(reward[0]) > 0:
                successes += 1
                
        success_rate = successes / self.n_eval_episodes
        
        self.logger.record("eval/success_rate", success_rate)
        self.logger.dump(self.num_timesteps)
        
        if success_rate >= self.success_threshold:
            self.timestep_to_threshold = self.num_timesteps
            print(f"Threshold reached at {self.num_timesteps} timesteps (success_rate={success_rate:.2f})")
            return False
        
        return True