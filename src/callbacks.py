import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import csv
from pathlib import Path
import time
from environments.adapter import EnvAdapter

class MiniGridCallback(BaseCallback):
    def __init__(
        self, 
        adapter, 
        eval_env, 
        eval_freq, 
        n_eval_episodes, 
        success_threshold, 
        csv_path: Path
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_threshold = success_threshold
        self.timestep_to_threshold = None
        self.csv_path = csv_path
        self.start_time = None
        self.adapter = adapter
        
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
            # TODO - implement is_success for crafter
            if self.adapter.is_success(float(reward[0]), info[0]):
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
    

class CrafterCallback(BaseCallback):
    """Training and evaluation callback for the Crafter environment.
    
    Logs per-episode training metrics (return, achievements, termination reason)
    and periodically runs deterministic evaluation episodes to compute the
    Crafter score (geometric mean of per-achievement success rates).
    
    When using implicit reward mode, self.locals["rewards"] is always 0.0
    during data collection (real rewards are assigned retroactively by the buffer).
    The _get_train_return() method handles this by reading the cumulative
    episode score from the reward model instead of the running reward accumulator.
    """
    def __init__(
        self, 
        adapter: EnvAdapter,
        eval_env, 
        eval_freq: int, 
        n_eval_episodes: int, 
        train_episode_csv_path: Path, 
        train_achievements_csv_path: Path, 
        eval_csv_path: Path,
        reward_model=None,
    ):
            super().__init__()
            self.adapter = adapter
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes
            self.train_episode_csv_path = Path(train_episode_csv_path)
            self.train_achievements_csv_path = Path(train_achievements_csv_path)
            self.eval_csv_path = Path(eval_csv_path)
            self.reward_model = reward_model
            
            self.start_time = None
            self.episode_id = 0
            self._running_train_return = 0.0
            self._running_episode_len = 0
        
    def _on_training_start(self) -> None:
        # Sanity check
        if self.training_env.num_envs != 1:
            raise ValueError(f"CrafterCallback supports num_envs == 1, got {self.training_env.num_envs}")
        
        self.start_time = time.time()
        
        # Ensure dirs exist
        self.train_episode_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.train_achievements_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        achievement_cols = self.adapter.achievement_column_names()
        success_rate_cols = self.adapter.success_rate_column_names()
        
        # 1) Compact training log
        with open(self.train_episode_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_id",
                "timestep",
                "episode_len",
                "train_episode_return",
                "num_achievements_unlocked",
                "termination_reason",
            ])
            
        # 2) Detailed training achievements log
        with open(self.train_achievements_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_id",
                "timestep",
                *achievement_cols,
            ])
            
        # 3) Eval metrics log
        with open(self.eval_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestep",
                "n_eval_episodes",
                "crafter_score",
                "mean_achievements_per_episode",
                "env_mean_return_per_episode",
                *success_rate_cols,
            ])
            
    def _termination_reason(self, info: dict) -> str:
        reason = info.get("termination_reason")
        if reason not in ("timeout", "died"):
            raise ValueError(
                f"Missing/invalid termination_reason in info: {reason!r}. "
                "Expected 'timeout' or 'died'."
            )
        return reason
    
    def _get_train_return(self) -> float:
        """Get the training return for the just-completed episode.
        
        In implicit mode, step rewards are always 0.0 (assigned retroactively
        by the buffer), so _running_train_return would always be 0. Instead,
        read the sum of segment scores from the reward model.
        In ground truth mode, use the standard running accumulator.
        """
        if self.reward_model is not None:
            return self.reward_model._last_episode_score
        return self._running_train_return
        
    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info = self.locals["infos"][0]
        
        self._running_train_return += reward
        self._running_episode_len += 1
        
        if done:
            train_return = self._get_train_return()
            
            achievements = dict(info.get("achievements", {}))
            bits = self.adapter.achievements_binary(achievements)
            num_unlocked = int(sum(bits))
            term_reason = self._termination_reason(info)
            
            # Write episode row in compact training log
            with open(self.train_episode_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_id,
                    self.num_timesteps,
                    self._running_episode_len,
                    round(train_return, 6),
                    num_unlocked,
                    term_reason,
                ])
                
            # Write training achievements log
            with open(self.train_achievements_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_id,
                    self.num_timesteps,
                    *bits,
                ])
                
            # TensorBoard training logs
            self.logger.record("train/train_episode_return", train_return)
            self.logger.record("train/episode_len", self._running_episode_len)
            self.logger.record("train/num_achievements_unlocked", num_unlocked)
            self.logger.dump(self.num_timesteps)
            
            # Reset per-episode accumulators
            self.episode_id += 1
            self._running_train_return = 0.0
            self._running_episode_len = 0
            
        # Eval loop trigger
        if self.num_timesteps % self.eval_freq == 0:
            self._run_eval()

        return True
    
    def _run_eval(self) -> None:
        """
        Run deterministic evaluation on eval_env (ground-truth reward env),
        aggregate Crafter metrics, and append one row to eval CSV.
        """
        achievement_names = self.adapter.achievement_names
        n_achievements = len(achievement_names)
        
        episode_returns = []
        episode_unlock_counts = []
        # Shape: [n_eval_episodes, 22]
        episode_bits = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()
            done = False
            ep_return = 0.0
            final_info = {}
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                
                ep_return += float(reward[0])
                done = bool(dones[0])
                final_info = infos[0]
                
            achievements = dict(final_info.get("achievements", {}))
            bits = self.adapter.achievements_binary(achievements)
            
            # Defensive check to avoid silent schema mismatches
            if len(bits) != n_achievements:
                raise ValueError(
                    f"Expected {n_achievements} achievement bits, got {len(bits)}"
                )
                
            episode_returns.append(ep_return)
            episode_unlock_counts.append(int(sum(bits)))
            episode_bits.append(bits)
            
        bits_arr = np.array(episode_bits, dtype=float) # [E, 22]
        success_rates = 100.0 * bits_arr.mean(axis=0) # percent space [0, 100]
        crafter_score = self._compute_crafter_score(success_rates)
        
        mean_achievements = float(np.mean(episode_unlock_counts))
        mean_env_return = float(np.mean(episode_returns))
        
        # TensorBoard eval logs
        self.logger.record("eval/crafter_score", crafter_score)
        self.logger.record("eval/mean_achievements_per_episode", mean_achievements)
        self.logger.record("eval/env_mean_return_per_episode", mean_env_return)
        for name, rate in zip(achievement_names, success_rates):
            self.logger.record(f"eval/success_rate/{name}", float(rate))
        self.logger.dump(self.num_timesteps)
        
        # CSV row: fixed order from schema
        with open(self.eval_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.num_timesteps,
                self.n_eval_episodes,
                round(float(crafter_score), 6),
                round(mean_achievements, 6),
                round(mean_env_return, 6),
                *[round(float(x), 6) for x in success_rates],
            ])
            
    def _compute_crafter_score(self, success_rates: np.ndarray) -> float:
        """Crafter score: geometric mean of (1 + success_rate_i) - 1, in percent space.
        
        This is the standard Crafter benchmark metric from Hafner (2022).
        Uses log-space computation for numerical stability.
        Input success_rates are in [0, 100] (percent).
        """
        success_rates = np.asarray(success_rates, dtype=float)
        if success_rates.ndim != 1:
            raise ValueError("success_rates must be a 1D array")
        if np.any(success_rates < 0.0) or np.any(success_rates > 100.0):
            raise ValueError("success_rates must be in [0, 100]")
        
        return float(np.exp(np.mean(np.log(1.0 + success_rates))) - 1.0)