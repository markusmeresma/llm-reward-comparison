from unittest.mock import MagicMock
from rewards import ImplicitRewardModel


def test_compute_reward_always_returns_zero():
    """compute_reward returns 0.0 on every step — real rewards are
    assigned retroactively by SegmentRolloutBuffer."""
    mock_client = MagicMock()
    mock_adapter = MagicMock()
    model = ImplicitRewardModel(mock_client, "task prompt", mock_adapter, segment_length=4)

    step_state = {"action": 2, "pos": (2, 1), "dir": 0}
    result = model.compute_reward(None, 0, None, 0.0,
                                  terminated=False, truncated=False, step_state=step_state)

    assert result == 0.0
    mock_client.evaluate_segment.assert_not_called()


def test_segment_evaluated_when_full():
    """LLM is called once the accumulator reaches segment_length steps."""
    mock_client = MagicMock()
    mock_client.evaluate_segment.return_value = (0.7, "good progress")
    mock_adapter = MagicMock()
    mock_adapter.segment_to_text.return_value = "Segment: 2 steps"
    model = ImplicitRewardModel(mock_client, "task prompt", mock_adapter, segment_length=2)

    step_a = {"action": 1, "pos": (1, 1), "dir": 0}
    step_b = {"action": 2, "pos": (1, 2), "dir": 0}

    model.compute_reward(None, 1, None, 0.0, terminated=False, truncated=False, step_state=step_a)
    model.compute_reward(None, 2, None, 0.0, terminated=False, truncated=False, step_state=step_b)

    mock_client.evaluate_segment.assert_called_once()
    pending = model.drain_pending()
    assert len(pending) == 1
    assert pending[0].score == 0.7
    assert pending[0].length == 2


def test_partial_segment_evaluated_on_episode_end():
    """Episode ending mid-segment triggers evaluation of the partial segment."""
    mock_client = MagicMock()
    mock_client.evaluate_segment.return_value = (0.3, "died early")
    mock_adapter = MagicMock()
    mock_adapter.segment_to_text.return_value = "Segment: 1 steps"
    model = ImplicitRewardModel(mock_client, "task prompt", mock_adapter, segment_length=4)

    step = {"action": 0, "pos": (1, 1), "dir": 0}
    result = model.compute_reward(None, 0, None, 0.0,
                                  terminated=True, truncated=False, step_state=step)

    assert result == 0.0
    mock_client.evaluate_segment.assert_called_once()
    pending = model.drain_pending()
    assert len(pending) == 1
    assert pending[0].length == 1


def test_last_episode_score_tracks_sum():
    """_last_episode_score holds the sum of segment scores for the completed episode."""
    mock_client = MagicMock()
    mock_client.evaluate_segment.side_effect = [(0.5, "ok"), (0.3, "died")]
    mock_adapter = MagicMock()
    mock_adapter.segment_to_text.return_value = "segment text"
    model = ImplicitRewardModel(mock_client, "task prompt", mock_adapter, segment_length=1)

    step_a = {"action": 1, "pos": (1, 1), "dir": 0}
    step_b = {"action": 0, "pos": (1, 2), "dir": 0}

    model.compute_reward(None, 1, None, 0.0, terminated=False, truncated=False, step_state=step_a)
    model.compute_reward(None, 0, None, 0.0, terminated=True, truncated=False, step_state=step_b)

    assert model._last_episode_score == 0.5 + 0.3
