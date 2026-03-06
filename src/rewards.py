from gymnasium import Wrapper
from abc import ABC, abstractmethod
from implicit_reward import build_segment_implicit_prompt
from dataclasses import dataclass
from segment import SegmentAccumulator
import logging
import importlib.util

class RewardModel(ABC):
    """Abstract interface for reward computation.
    
    All reward paradigms (ground truth, implicit LLM, explicit LLM) implement
    this interface, allowing the training pipeline to swap reward models without
    code changes. The wrapper calls compute_reward() on every env step.
    """
    @abstractmethod
    def compute_reward(self, state, action, next_state, reward,
                       terminated: bool, truncated: bool, step_state: dict) -> float:
        pass

class GroundTruthRewardModel(RewardModel):
    """Passes through the environment's native reward unchanged (baseline)."""
    def compute_reward(self, state, action, next_state, reward,
                       terminated, truncated, step_state) -> float:
        return reward
    
@dataclass
class PendingSegment:
    """A scored segment awaiting retroactive reward assignment by the buffer."""
    score: float
    length: int
    reasoning: str

class ImplicitRewardModel(RewardModel):
    """Segment-based implicit LLM reward model (Kwon et al. 2023 style).
    
    Accumulates step-state dicts and evaluates them in segments of K steps via
    LLM. Always returns 0.0 from compute_reward() — actual rewards are written
    retroactively by SegmentRolloutBuffer.compute_returns_and_advantage().
    
    Segments are evaluated when:
      - The accumulator reaches K steps.
      - An episode ends (partial segment).
      - The rollout buffer calls flush_segment() at rollout boundary.
    
    Episode-level scores are tracked for callback logging: _current_episode_score
    accumulates segment scores within an episode, and _last_episode_score stores
    the total for the most recently completed episode.
    """
    def __init__(self, llm_client, task_prompt: str, adapter, segment_length: int) -> None:
        self.llm_client = llm_client
        self.task_prompt = task_prompt
        self.adapter = adapter
        self._accumulator = SegmentAccumulator(segment_length)
        self._pending_segments: list[PendingSegment] = []
        self._current_episode_score: float = 0.0
        self._last_episode_score: float = 0.0
    
    def compute_reward(self, state, action, next_state, reward,
                       terminated, truncated, step_state) -> float:
        """Process one environment step. Returns 0.0 always — real LLM rewards
        are assigned retroactively by SegmentRolloutBuffer."""
        self._accumulator.add_step(step_state)
        
        episode_ended = terminated or truncated
        if episode_ended:
            reason = "timeout" if truncated else "died"
            self._accumulator.mark_episode_end(reason)
            
        if self._accumulator.is_full() or episode_ended:
            result = self._accumulator.flush()
            if result is not None:
                pending = self._evaluate_segment(result)
                self._pending_segments.append(pending)
                self._current_episode_score += pending.score
                
        if episode_ended:
            self._last_episode_score = self._current_episode_score
            self._current_episode_score = 0.0
            self._accumulator.reset_for_new_episode()
        
        return 0.0
    
    def _evaluate_segment(self, result) -> PendingSegment:
        """Convert segment to text, query LLM, return scored PendingSegment."""
        segment_text = self.adapter.segment_to_text(result)
        prompt = build_segment_implicit_prompt(self.task_prompt, segment_text)
        score, reasoning = self.llm_client.evaluate_segment(prompt, len(result.steps))
        return PendingSegment(score=score, length=len(result.steps), reasoning=reasoning)
    
    def flush_segment(self) -> None:
        """Force-evaluate any in-progress partial segment (called at rollout boundary)."""
        result = self._accumulator.flush()
        if result is not None:
            pending = self._evaluate_segment(result)
            self._pending_segments.append(pending)
            self._current_episode_score += pending.score
            
    def drain_pending(self) -> list[PendingSegment]:
        """Return and clear all pending segments. Called by SegmentRolloutBuffer
        to retrieve scored segments for retroactive reward assignment."""
        pending = self._pending_segments
        self._pending_segments = []
        return pending
        
    
class RewardModelWrapper(Wrapper):
    """Gymnasium wrapper that intercepts env rewards and delegates to a
    pluggable RewardModel.
    
    Transparent to the RL algorithm — it sees the same obs/done/info, but
    receives the reward model's output instead of the environment's native reward.
    """
    def __init__(self, env, reward_model: RewardModel, adapter):
        super().__init__(env)
        self.reward_model = reward_model
        self.adapter = adapter
        self._current_obs = None
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_obs = obs
        # Side effect: adapters cache env state on reset (e.g. MiniGrid goal position).
        self.adapter.extract_initial_state(self.env)
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
        
        new_reward = self.reward_model.compute_reward(
            state=self._current_obs,  
            action=action,
            next_state=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            step_state=step_state,
        )
        self._current_obs = obs
        return obs, new_reward, terminated, truncated, info
    
class ExplicitRewardModel(RewardModel):
    """Explicit LLM reward model (EUREKA-style).

    Loads a pre-generated Python file containing a compute_reward() function
    and calls it on every environment step. No LLM calls at runtime - the
    generated code runs locally as a normal Python function.

    The generated function signature is:
        compute_reward(current_step, prev_step, terminated, truncated) -> float

    where current_step and prev_step are the step_state dicts produced by the
    environment adapter. prev_step is None on the first step of each episode.

    This class tracks prev_step internally and resets it to None when an
    episode ends (terminated or truncated), so the generated function always
    receives the correct previous state.
    """
    def __init__(self, reward_code_path: str) -> None:
        """Load the generated reward function from a .py file.

        Args:
            reward_code_path: Path to the generated reward_fn.py file.
                Loaded via importlib so tracebacks show real file paths
                and line numbers (unlike exec() which shows '<string>').

        Raises:
            FileNotFoundError: If reward_code_path does not exist.
            AttributeError: If the loaded module has no compute_reward function.
        """
        self._logger = logging.getLogger(__name__)
        self._prev_step = None
        
        # importlib.util.spec_from_file_location creates a module spec from a
        # file path. We then create the module object and execute it, which is
        # equivalent to "import reward_fn" but from an arbitrary file path.
        spec = importlib.util.spec_from_file_location("reward_fn", reward_code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract the function the LLM was instructed to define
        if not hasattr(module, "compute_reward"):
            raise AttributeError(
                f"Generated module {reward_code_path} has no 'compute_reward' function"
            )
            
        self._reward_fn = module.compute_reward
        self._logger.info(f"Loaded explicit reward function from {reward_code_path}")
        
    def compute_reward(self, state, action, next_state, reward,
                       terminated, truncated, step_state) -> float:
        """Call the generated reward function with the current and previous step states.

        Args:
            state: Raw observation (unused — the generated function works on step_state).
            action: Action taken (unused — already encoded in step_state).
            next_state: Next raw observation (unused).
            reward: Environment's native reward (unused — explicit replaces it entirely).
            terminated: True if the episode ended naturally.
            truncated: True if the episode hit the time limit.
            step_state: Dict from adapter.extract_step_state() for this step.

        Returns:
            Scalar reward from the generated function. On crash, returns 0.0
            so training continues (same fallback strategy as ImplicitRewardModel).
        """
        try:
            result = self._reward_fn(
                current_step=step_state,
                prev_step=self._prev_step,
                terminated=terminated,
                truncated=truncated,
            )
        except Exception as e:
            # Generated code can crash on edge cases (e.g. missing dict keys,
            # division by zero). Log the error and return neutral reward so
            # training doesn't abort.
            self._logger.warning(f"Explicit reward function crashed: {e}")
            result = 0.0
            
        self._prev_step = step_state
        
        # When the episode ends, reset prev_step so the next episode's first
        # step correctly receives prev_step=None.
        if terminated or truncated:
            self._prev_step = None

        return float(result)