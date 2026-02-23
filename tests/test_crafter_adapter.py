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


def _make_inv(**overrides):
    """Helper: Crafter inventory with sensible defaults."""
    inv = {
        "health": 9, "food": 9, "drink": 9, "energy": 9,
        "wood": 0, "stone": 0, "coal": 0, "iron": 0, "diamond": 0, "sapling": 0,
        "wood_pickaxe": 0, "stone_pickaxe": 0, "iron_pickaxe": 0,
        "wood_sword": 0, "stone_sword": 0, "iron_sword": 0,
    }
    inv.update(overrides)
    return inv


def test_segment_to_text_ongoing_episode():
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 1,  # move_left
                "pos": (1, 1),
                "inventory": _make_inv(),
                "achievements": {"collect_wood": 0},
            },
            {
                "action": 5,  # do (interact) -> collects wood
                "pos": (1, 2),
                "inventory": _make_inv(food=8, wood=1),
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
    assert "Failed" not in text


def test_segment_to_text_episode_ended():
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 0,  # noop
                "pos": (3, 3),
                "inventory": _make_inv(health=1, food=0, drink=0, energy=0),
                "achievements": {},
            },
            {
                "action": 0,  # noop
                "pos": (3, 3),
                "inventory": _make_inv(health=0, food=0, drink=0, energy=0),
                "achievements": {},
            },
        ],
        episode_ended=True,
        termination_reason="died",
    )

    text = adapter.segment_to_text(result)

    assert "episode ended (died)" in text
    assert "health 1→0 (-1)" in text


def test_segment_to_text_successful_craft():
    """A craft action that results in inventory change is reported as successful."""
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 1,  # move_left
                "pos": (2, 2),
                "inventory": _make_inv(wood=2),
                "achievements": {},
            },
            {
                "action": 11,  # make_wood_pickaxe -> succeeds
                "pos": (2, 2),
                "inventory": _make_inv(wood=1, wood_pickaxe=1),
                "achievements": {"make_wood_pickaxe": 1},
            },
        ],
        episode_ended=False,
        termination_reason=None,
    )

    text = adapter.segment_to_text(result)

    assert "Successful crafts/placements: 1 make_wood_pickaxe" in text
    assert "Failed" not in text


def test_segment_to_text_failed_crafts():
    """Craft actions with no inventory change are counted as failed attempts."""
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 1,  # move_left
                "pos": (2, 2),
                "inventory": _make_inv(),
                "achievements": {},
            },
            {
                "action": 13,  # make_iron_pickaxe -> fails (no materials)
                "pos": (2, 2),
                "inventory": _make_inv(),
                "achievements": {},
            },
            {
                "action": 16,  # make_iron_sword -> fails
                "pos": (2, 2),
                "inventory": _make_inv(),
                "achievements": {},
            },
        ],
        episode_ended=False,
        termination_reason=None,
    )

    text = adapter.segment_to_text(result)

    assert "Successful" not in text
    assert "Failed craft/place attempts: 2" in text


def test_segment_to_text_mixed_success_and_failure():
    """Mix of successful and failed craft/place actions."""
    adapter = CrafterAdapter()
    result = SegmentResult(
        steps=[
            {
                "action": 1,  # move_left
                "pos": (2, 2),
                "inventory": _make_inv(wood=3, sapling=1),
                "achievements": {},
            },
            {
                "action": 11,  # make_wood_pickaxe -> succeeds
                "pos": (2, 2),
                "inventory": _make_inv(wood=2, sapling=1, wood_pickaxe=1),
                "achievements": {},
            },
            {
                "action": 13,  # make_iron_pickaxe -> fails
                "pos": (2, 2),
                "inventory": _make_inv(wood=2, sapling=1, wood_pickaxe=1),
                "achievements": {},
            },
            {
                "action": 10,  # place_plant -> succeeds (sapling consumed)
                "pos": (2, 3),
                "inventory": _make_inv(wood=2, sapling=0, wood_pickaxe=1),
                "achievements": {},
            },
        ],
        episode_ended=False,
        termination_reason=None,
    )

    text = adapter.segment_to_text(result)

    assert "Successful crafts/placements:" in text
    assert "make_wood_pickaxe" in text
    assert "place_plant" in text
    assert "Failed craft/place attempts: 1" in text
