from environments.minigrid_adapter import MiniGridAdapter
from models import Trajectory


def test_trajectory_to_text_formats_minigrid_episode():
    adapter = MiniGridAdapter()
    trajectory = Trajectory(
        initial_state={"pos": (1, 1), "dir": 0, "goal_pos": (3, 3)},
        steps=[
            {"action": 2, "pos": (1, 2), "dir": 0},
            {"action": 0, "pos": (1, 2), "dir": 3},
        ],
    )

    text = adapter.trajectory_to_text(
        trajectory, "MiniGrid-Empty-5x5-v0", terminated=True, truncated=False
    )

    assert "Environment: MiniGrid-Empty-5x5-v0" in text
    assert "Start: pos=(1, 1), dir=0" in text
    assert "Goal: pos=(3, 3)" in text
    assert " 0: forward -> pos=(1, 2)" in text
    assert " 1: turn_left -> pos=(1, 2)" in text


def test_is_success_depends_on_positive_reward():
    adapter = MiniGridAdapter()
    assert adapter.is_success(1.0, {}) is True
    assert adapter.is_success(0.0, {}) is False
