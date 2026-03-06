import pytest
from rewards import ExplicitRewardModel


def _write_reward_fn(tmp_path, body="return 1.0"):
    """Write a minimal reward_fn.py and return its path."""
    code = (
        f"def compute_reward(current_step, prev_step, terminated, truncated):\n"
        f"    {body}\n"
    )
    path = tmp_path / "reward_fn.py"
    path.write_text(code)
    return str(path)


def _step(model, step_state, terminated=False, truncated=False):
    """Shorthand for calling compute_reward with only the fields that matter."""
    return model.compute_reward(
        state=None, action=0, next_state=None, reward=0.0,
        terminated=terminated, truncated=truncated, step_state=step_state,
    )


class TestLoading:
    def test_loads_valid_reward_function(self, tmp_path):
        path = _write_reward_fn(tmp_path)
        model = ExplicitRewardModel(path)
        assert _step(model, {"x": 1}) == 1.0

    def test_raises_if_no_compute_reward(self, tmp_path):
        code = "def other_fn(): return 0\n"
        path = tmp_path / "reward_fn.py"
        path.write_text(code)
        with pytest.raises(AttributeError, match="no 'compute_reward'"):
            ExplicitRewardModel(str(path))


class TestDelegation:
    def test_passes_step_state_as_current_step(self, tmp_path):
        path = _write_reward_fn(tmp_path, "return current_step['val']")
        model = ExplicitRewardModel(path)
        assert _step(model, {"val": 3.5}) == 3.5

    def test_passes_terminated_and_truncated(self, tmp_path):
        path = _write_reward_fn(
            tmp_path, "return 10.0 if terminated else (5.0 if truncated else 0.0)"
        )
        model = ExplicitRewardModel(path)
        assert _step(model, {}, terminated=False, truncated=False) == 0.0
        assert _step(model, {}, terminated=True) == 10.0

    def test_result_is_always_float(self, tmp_path):
        path = _write_reward_fn(tmp_path, "return 3")
        model = ExplicitRewardModel(path)
        result = _step(model, {})
        assert isinstance(result, float)
        assert result == 3.0


class TestPrevStepTracking:
    def test_first_step_gets_prev_step_none(self, tmp_path):
        path = _write_reward_fn(
            tmp_path, "return 0.0 if prev_step is None else 1.0"
        )
        model = ExplicitRewardModel(path)
        assert _step(model, {"a": 1}) == 0.0

    def test_second_step_gets_previous_step_state(self, tmp_path):
        path = _write_reward_fn(
            tmp_path, "return prev_step['val'] if prev_step else -1.0"
        )
        model = ExplicitRewardModel(path)
        _step(model, {"val": 7.0})
        assert _step(model, {"val": 99.0}) == 7.0

    def test_prev_step_resets_after_terminated(self, tmp_path):
        path = _write_reward_fn(
            tmp_path, "return 0.0 if prev_step is None else 1.0"
        )
        model = ExplicitRewardModel(path)
        _step(model, {"a": 1})
        _step(model, {"a": 2}, terminated=True)
        assert _step(model, {"a": 3}) == 0.0

    def test_prev_step_resets_after_truncated(self, tmp_path):
        path = _write_reward_fn(
            tmp_path, "return 0.0 if prev_step is None else 1.0"
        )
        model = ExplicitRewardModel(path)
        _step(model, {"a": 1})
        _step(model, {"a": 2}, truncated=True)
        assert _step(model, {"a": 3}) == 0.0

    def test_multi_episode_sequence(self, tmp_path):
        path = _write_reward_fn(
            tmp_path, "return 0.0 if prev_step is None else 1.0"
        )
        model = ExplicitRewardModel(path)

        # Episode 1: 3 steps
        assert _step(model, {"a": 1}) == 0.0  # first step -> None
        assert _step(model, {"a": 2}) == 1.0  # has prev
        assert _step(model, {"a": 3}, terminated=True) == 1.0  # has prev, episode ends

        # Episode 2: prev_step should be None again
        assert _step(model, {"a": 4}) == 0.0
        assert _step(model, {"a": 5}, truncated=True) == 1.0

        # Episode 3
        assert _step(model, {"a": 6}) == 0.0


class TestCrashHandling:
    def test_crash_returns_zero(self, tmp_path):
        path = _write_reward_fn(tmp_path, "return 1 / 0")
        model = ExplicitRewardModel(path)
        assert _step(model, {"x": 1}) == 0.0

    def test_crash_on_missing_key_returns_zero(self, tmp_path):
        path = _write_reward_fn(tmp_path, "return current_step['missing']")
        model = ExplicitRewardModel(path)
        assert _step(model, {"x": 1}) == 0.0

    def test_prev_step_still_updated_after_crash(self, tmp_path):
        path = _write_reward_fn(
            tmp_path,
            "return prev_step['val'] if prev_step else current_step['missing']",
        )
        model = ExplicitRewardModel(path)
        # First step crashes (missing key), but prev_step should still be set
        assert _step(model, {"val": 5.0}) == 0.0
        # Second step should see prev_step correctly
        assert _step(model, {"val": 9.0}) == 5.0
