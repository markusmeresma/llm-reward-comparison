from dataclasses import dataclass, field

@dataclass
class SegmentResult:
    """Output of flushing a segment accumulator.
    
    Contains the raw step-state dicts collected over K steps (or fewer for
    partial segments at episode/rollout boundaries), plus metadata about
    whether the episode ended during this segment.
    """
    steps: list[dict]
    episode_ended: bool
    termination_reason: str | None
    
class SegmentAccumulator:
    """Collects step-state dicts and produces SegmentResults every K steps.
    
    Environment-agnostic: stores raw step dicts and delegates summary
    formatting to the adapter's segment_to_text(). Has no knowledge of
    rollout buffer positions — index mapping is handled by SegmentRolloutBuffer.
    
    Segments are flushed in two cases:
      1. The accumulator reaches K steps (is_full()).
      2. The episode ends (partial segment).
    At rollout boundaries, flush() is called externally by the buffer
    to evaluate any in-progress partial segment.
    """
    def __init__(self, segment_length: int):
        self.segment_length = segment_length
        self._steps: list[dict] = []
        self._episode_ended: bool = False
        self._termination_reason: str | None = None
        
    def add_step(self, step_state: dict) -> None:
        self._steps.append(step_state)
        
    def is_full(self) -> bool:
        return len(self._steps) >= self.segment_length
    
    def mark_episode_end(self, reason: str) -> None:
        self._episode_ended = True
        self._termination_reason = reason
        
    def flush(self) -> SegmentResult | None:
        """Return accumulated steps as a SegmentResult and reset internal state.
        Returns None if no steps have been accumulated (e.g. flush on empty accumulator).
        """
        if not self._steps:
            return None
        result = SegmentResult(
            steps=self._steps,
            episode_ended=self._episode_ended,
            termination_reason=self._termination_reason,
        )
        self._steps = []
        self._episode_ended = False
        self._termination_reason = None
        return result
    
    def reset_for_new_episode(self) -> None:
        """Discard any accumulated steps. Called after flushing on episode boundary."""
        self._steps = []
        self._episode_ended = False
        self._termination_reason = None
        
    