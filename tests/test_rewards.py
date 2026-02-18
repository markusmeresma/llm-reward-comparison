from unittest.mock import MagicMock
from rewards import ImplicitRewardModel
from models import Trajectory

def test_compute_reward_returns_zero_mid_episode():
    """No LLM call when episode is not done."""
    mock_client = MagicMock()
    mock_adapter = MagicMock()
    model = ImplicitRewardModel(mock_client, "MiniGrid-Empty-5x5-v0", "task prompt", mock_adapter)

    traj = Trajectory(initial_state={"pos": (1, 1), "dir": 0, "goal_pos": (3, 3)}, steps=[])
    result = model.compute_reward(None, 0, None, 0.0, terminated=False, truncated=False, trajectory=traj)

    assert result == 0.0
    mock_client.evaluate_trajectory.assert_not_called()
    mock_adapter.trajectory_to_text.assert_not_called()
    
def test_compute_reward_calls_llm_on_termination():
    """LLM is called when episode terminates."""
    mock_client = MagicMock()
    mock_adapter = MagicMock()
    mock_adapter.trajectory_to_text.return_value = "Environment: MiniGrid-Empty-5x5-v0"
    mock_client.evaluate_trajectory.return_value = 1.0
    model = ImplicitRewardModel(mock_client, "MiniGrid-Empty-5x5-v0", "task prompt", mock_adapter)

    traj = Trajectory(
        initial_state={"pos": (1, 1), "dir": 0, "goal_pos": (3, 3)},
        steps=[{"action": 2, "pos": (2, 1), "dir": 0}],
    )
    result = model.compute_reward(None, 2, None, 0.0, terminated=True, truncated=False, trajectory=traj)

    assert result == 1.0
    mock_adapter.trajectory_to_text.assert_called_once_with(
        traj, "MiniGrid-Empty-5x5-v0", True, False
    )
    mock_client.evaluate_trajectory.assert_called_once()
    
    prompt_arg = mock_client.evaluate_trajectory.call_args[0][0]
    assert "MiniGrid-Empty-5x5-v0" in prompt_arg
    assert "--- TRAJECTORY ---" in prompt_arg