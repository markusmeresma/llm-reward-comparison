from gymnasium import Wrapper
from abc import ABC, abstractmethod
from dataclasses import dataclass
from minigrid.core.grid import Grid

@dataclass
class Step:
    action: int
    pos: tuple
    dir: int

@dataclass
class Trajectory:
    initial_pos: tuple
    initial_dir: int
    goal_pos: tuple
    steps: list[Step]
    outcome: dict = None
    
def get_goal_pos(grid: Grid) -> tuple:
    for x in range(grid.width):
        for y in range(grid.height):
            obj = grid.get(x, y)
            if obj is not None and obj.type == "goal":
                goal_pos = (x, y)
                return goal_pos

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
        self.trajectory = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        self.trajectory = Trajectory(
            initial_pos=self.env.unwrapped.agent_pos,
            initial_dir=self.env.unwrapped.agent_dir,
            goal_pos=get_goal_pos(self.env.unwrapped.grid),
            steps=[]
        )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.trajectory.steps.append(Step(
            action=action,
            pos=self.env.unwrapped.agent_pos,
            dir=self.env.unwrapped.agent_dir
        ))
        new_reward = self.reward_model.compute_reward(
            state=self._current_obs,  
            action=action,
            next_state=obs,
            reward=reward,
        )
        self._current_obs = obs
        return obs, new_reward, terminated, truncated, info