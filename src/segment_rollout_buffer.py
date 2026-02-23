from stable_baselines3.common.buffers import RolloutBuffer

class SegmentRolloutBuffer(RolloutBuffer):
    """Custom RolloutBuffer that retroactively writes LLM segment scores into
    step rewards before GAE computation.
    
    Why a custom buffer instead of a callback or PPO subclass:
    SB3's collect_rollouts() calls compute_returns_and_advantage() *before*
    on_rollout_end(), so a callback would be too late. A PPO subclass would
    require copying ~50 lines of collect_rollouts(). Overriding
    compute_returns_and_advantage() is the cleanest insertion point — rewards
    are modified inside the method that computes GAE, guaranteeing they're used.
    
    Requires n_envs=1 because segment tracking assumes a single environment.
    """
    def __init__(self, *args, reward_model, min_segment_length=16, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError(
                f"SegmentRolloutBuffer requires n_envs=1, got {self.n_envs}"
            )
        self.reward_model = reward_model
        self.min_segment_length = min_segment_length
        
    def compute_returns_and_advantage(self, last_values, dones):
        """Write segment scores into step rewards, then compute GAE.
    
        Each step in a segment receives score/divisor, where divisor is the
        segment's actual length (floored at min_segment_length to prevent
        extreme per-step values from very short partial segments). This makes
        the total reward across a segment equal to the LLM score, regardless
        of segment length — K only controls how often the LLM is called, not
        reward magnitude.
        
        Uses += to preserve SB3's timeout bootstrapping (gamma * V(terminal_obs))
        already written at truncation steps during collect_rollouts().
        """
        self.reward_model.flush_segment()
        pending = self.reward_model.drain_pending()
        
        # Reconstruct buffer positions from ordered segment lengths and write
        # uniform per-step rewards. Segments are contiguous and in-order by
        # construction, so cumulative sum of lengths maps directly to buffer indices.
        pos = 0
        for seg in pending:
            # Floor prevents extreme per-step values from very short partial
            # segments (e.g. 1-3 steps at rollout boundaries).
            divisor = max(seg.length, self.min_segment_length)
            # += (not =) preserves SB3's timeout bootstrapping value
            # (gamma * V(terminal_obs)) written at truncation steps.
            self.rewards[pos : pos + seg.length, 0] += seg.score / divisor
            pos += seg.length
        assert pos == self.buffer_size, (
            f"Segments covered {pos} steps, expected {self.buffer_size}"
        )
        
        super().compute_returns_and_advantage(last_values, dones)