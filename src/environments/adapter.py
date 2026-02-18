from abc import ABC, abstractmethod
import gymnasium as gym
from models import Trajectory

class EnvAdapter(ABC):
    @abstractmethod
    def make_base_env(self, env_id: str, seed: int = None, render_mode=None) -> gym.Env:
        """Create the base Gymnasium env with correct obs wrappers applied."""
        pass
    
    @abstractmethod
    def extract_initial_state(self, env: gym.Env) -> dict:
        """Extract state info on reset (for trajectory tracking)."""
        pass
    
    @abstractmethod
    def extract_step_state(self, env: gym.Env, action: int, info: dict) -> dict:
        """Extract state info after a step (for trajectory tracking)."""
        pass
    
    @abstractmethod
    def trajectory_to_text(self, trajectory: Trajectory, env_id: str, terminated: bool, truncated: bool) -> str:
        """Convert trajectory to text for LLM evaluation."""
        pass
    
    @abstractmethod
    def is_success(self, reward: float, info: dict) -> bool:
        """Determine if an episode was successful (for eval callback)."""
        pass
    
    @property
    @abstractmethod
    def action_names(self) -> dict[int, str]:
        """Mapping from action int to human-readable name."""
        pass