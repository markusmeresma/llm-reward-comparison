from environments.crafter_adapter import CrafterAdapter
from models import Trajectory


def test_extract_initial_state_is_empty_dict():
    adapter = CrafterAdapter()
    assert adapter.extract_initial_state(env=None) == {}


def test_extract_step_state_uses_info_payload():
    adapter = CrafterAdapter()
    info = {
        "player_pos": [4, 7],
        "inventory": {"wood": 2},
        "achievements": {"collect_wood": 1},
    }

    state = adapter.extract_step_state(env=None, action=3, info=info)

    assert state["action"] == 3
    assert state["pos"] == (4, 7)
    assert state["inventory"] == {"wood": 2}
    assert state["achievements"] == {"collect_wood": 1}


def test_trajectory_to_text():
    adapter = CrafterAdapter()
    trajectory = Trajectory(
        initial_state={},
        steps=[
            {
                "action": 1,
                "pos": (1, 1),
                "inventory": {"wood": 1},
                "achievements": {"collect_wood": 0},
            },
            {
                "action": 5,
                "pos": (1, 2),
                "inventory": {"wood": 2},
                "achievements": {"collect_wood": 1},
            },
        ],
    )

    text = adapter.trajectory_to_text(trajectory, "CrafterReward-v1", True, False)

    assert "Environment: CrafterReward-v1" in text
    assert "Final inventory: {'wood': 2}" in text
    assert "Achievements unlocked: ['collect_wood']" in text
    assert " 0: move_left -> pos=(1, 1)" in text
    assert " 1: do -> pos=(1, 2)" in text

