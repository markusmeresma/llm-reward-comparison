from environments.minigrid_adapter import MiniGridAdapter
from segment import SegmentResult


def test_segment_to_text_ongoing_episode():
    adapter = MiniGridAdapter()
    adapter._goal_pos = (3, 3)
    result = SegmentResult(
        steps=[
            {"action": 2, "pos": (1, 2), "dir": 0},
            {"action": 0, "pos": (1, 2), "dir": 3},
        ],
        episode_ended=False,
        termination_reason=None,
    )

    text = adapter.segment_to_text(result)

    assert "Segment: 2 steps (episode ongoing)" in text
    assert "goal=(3, 3)" in text
    assert "Reached goal" not in text


def test_segment_to_text_reached_goal():
    adapter = MiniGridAdapter()
    adapter._goal_pos = (3, 3)
    result = SegmentResult(
        steps=[
            {"action": 2, "pos": (2, 3), "dir": 0},
            {"action": 2, "pos": (3, 3), "dir": 0},
        ],
        episode_ended=True,
        termination_reason=None,
    )

    text = adapter.segment_to_text(result)

    assert "episode ended" in text
    assert "Reached goal: yes" in text


def test_is_success_depends_on_positive_reward():
    adapter = MiniGridAdapter()
    assert adapter.is_success(1.0, {}) is True
    assert adapter.is_success(0.0, {}) is False
