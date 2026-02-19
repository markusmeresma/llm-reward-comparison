from gymnasium import Wrapper
from abc import ABC, abstractmethod
from implicit_reward import build_binary_implicit_prompt
from models import Trajectory

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
    def __init__(self, llm_client, env_id: str, task_prompt: str, adapter) -> None:
        self.llm_client = llm_client
        self.env_id = env_id
        self.task_prompt = task_prompt
        self.adapter = adapter
    
    def compute_reward(self, state, action, next_state, reward,
                       terminated, truncated, trajectory) -> float:
        # Only call LLM at episode end
        if not (terminated or truncated):
            return 0.0
        
        traj_text = self.adapter.trajectory_to_text(trajectory, self.env_id, terminated, truncated)
        prompt = build_binary_implicit_prompt(self.task_prompt, traj_text)
        return self.llm_client.evaluate_trajectory(prompt)
    
class RewardModelWrapper(Wrapper):
    def __init__(self, env, reward_model: RewardModel, adapter):
        super().__init__(env)
        self.reward_model = reward_model
        self.adapter = adapter
        self._current_obs = None
        self.trajectory = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        initial_state = self.adapter.extract_initial_state(self.env)
        self.trajectory = Trajectory(initial_state=initial_state)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Attach explicit episode-end reason for downstream logging.
        info = dict(info or {})
        if truncated:
            info["termination_reason"] = "timeout"
        elif terminated:
            info["termination_reason"] = "died"
        
        step_state = self.adapter.extract_step_state(self.env, action, info)
        self.trajectory.steps.append(step_state)
        
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