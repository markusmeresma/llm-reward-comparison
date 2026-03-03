from abc import ABC, abstractmethod
import gymnasium as gym
from segment import SegmentResult

class EnvAdapter(ABC):
    """Abstract adapter for environment-specific operations.
    
    Each environment (MiniGrid, Crafter) has its own adapter that handles
    env creation, state extraction for segment accumulation, and conversion
    of segment data to text summaries for LLM evaluation.
    """
    @abstractmethod
    def make_base_env(self, env_id: str, seed: int = None, render_mode=None) -> gym.Env:
        """Create the base Gymnasium env with correct obs wrappers applied."""
        pass
    
    def extract_initial_state(self, env: gym.Env) -> dict:
        """Called on env reset. Adapters can cache env state here (e.g. goal position).
        Default is a no-op."""
        return {}
    
    @abstractmethod
    def extract_step_state(self, env: gym.Env, action: int, info: dict) -> dict:
        """Extract per-step state dict passed to the reward model and segment accumulator."""
        pass
    
    @abstractmethod
    def segment_to_text(self, result: SegmentResult) -> str:
        """Convert a SegmentResult (K steps of raw data) into a compact text
        summary for LLM evaluation. Format is environment-specific."""
        pass
    
    @property
    @abstractmethod
    def action_names(self) -> dict[int, str]:
        """Mapping from action int to human-readable name."""
        pass