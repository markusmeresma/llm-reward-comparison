from gymnasium import Wrapper
from abc import ABC, abstractmethod

class RewardModel(ABC):
    @abstractmethod
    def compute_reward(self, state, action, next_state, reward) -> float:
        pass

class GroundTruthRewardModel(RewardModel):
    def compute_reward(self, state, action, next_state, reward) -> float:
        return reward

class ImplicitRewardModel(RewardModel):
    def compute_reward(self, state, action, next_state, reward) -> float:
        # todo - use LLM to compute reward
        return reward
    
class RewardModelWrapper(Wrapper):
    def __init__(self, env, reward_model: RewardModel):
        super().__init__(env)
        self.reward_model = reward_model
        self._current_obs = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_reward = self.reward_model.compute_reward(
            state=self._current_obs,  
            action=action,
            next_state=obs,
            reward=reward,
        )
        self._current_obs = obs
        return obs, new_reward, terminated, truncated, info