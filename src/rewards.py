from gymnasium import Wrapper
from abc import ABC, abstractmethod
from implicit_reward import build_binary_implicit_prompt, trajectory_to_text
from models import Trajectory, get_goal_pos, Step

class RewardModel(ABC):
    @abstractmethod
    def compute_reward(self, state, action, next_state, reward,
                       terminated: bool, truncated: bool, trajectory) -> float:
        pass

class GroundTruthRewardModel(RewardModel):
    def compute_reward(self, state, action, next_state, reward,
                       terminated, truncated, trajectory) -> float:
        return reward

class ImplicitRewardModel(RewardModel):
    def __init__(self, llm_client, env_id: str, task_prompt: str) -> None:
        self.llm_client = llm_client
        self.env_id = env_id
        self.task_prompt = task_prompt
    
    def compute_reward(self, state, action, next_state, reward,
                       terminated, truncated, trajectory) -> float:
        # Only call LLM at episode end
        if not (terminated or truncated):
            return 0.0
        
        traj_text = trajectory_to_text(trajectory, self.env_id, terminated, truncated)
        prompt = build_binary_implicit_prompt(self.task_prompt, traj_text)
        return self.llm_client.evaluate_trajectory(prompt)
    
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
            terminated=terminated,
            truncated=truncated,
            trajectory=self.trajectory,
        )
        self._current_obs = obs
        return obs, new_reward, terminated, truncated, info