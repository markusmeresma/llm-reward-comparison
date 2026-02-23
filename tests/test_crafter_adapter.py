from environments.crafter_adapter import CrafterAdapter
from segment import SegmentResult


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


def test_segment_to_text_ongoing_episode():
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 1,
                "pos": (1, 1),
                "inventory": {"health": 9, "food": 9, "drink": 9, "energy": 9, "wood": 0},
                "achievements": {"collect_wood": 0},
            },
            {
                "action": 5,
                "pos": (1, 2),
                "inventory": {"health": 9, "food": 8, "drink": 9, "energy": 9, "wood": 1},
                "achievements": {"collect_wood": 1},
            },
        ],
        episode_ended=False,
        termination_reason=None,
    )

    text = adapter.segment_to_text(result)

    assert "Segment: 2 steps (episode ongoing)" in text
    assert "collect_wood" in text
    assert "+1 wood" in text


def test_segment_to_text_episode_ended():
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 0,
                "pos": (3, 3),
                "inventory": {"health": 1, "food": 0, "drink": 0, "energy": 0},
                "achievements": {},
            },
            {
                "action": 0,
                "pos": (3, 3),
                "inventory": {"health": 0, "food": 0, "drink": 0, "energy": 0},
                "achievements": {},
            },
        ],
        episode_ended=True,
        termination_reason="died",
    )

    text = adapter.segment_to_text(result)

    assert "episode ended (died)" in text
    assert "health 1→0 (-1)" in text
