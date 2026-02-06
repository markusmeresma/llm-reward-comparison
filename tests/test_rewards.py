from unittest.mock import MagicMock
from rewards import ImplicitRewardModel, Trajectory, Step

def test_compute_reward_returns_zero_mid_episode():
    """No LLM call when episode is not done."""
    mock_client = MagicMock()
    model = ImplicitRewardModel(mock_client, "MiniGrid-Empty-5x5-v0", "task prompt")

    traj = Trajectory(initial_pos=(1,1), initial_dir=0, goal_pos=(3,3), steps=[])
    result = model.compute_reward(None, 0, None, 0.0, terminated=False, truncated=False, trajectory=traj)

    assert result == 0.0
    mock_client.evaluate_trajectory.assert_not_called()
    
def test_compute_reward_calls_llm_on_termination():
    """LLM is called when episode terminates."""
    mock_client = MagicMock()
    mock_client.evaluate_trajectory.return_value = 1.0
    model = ImplicitRewardModel(mock_client, "MiniGrid-Empty-5x5-v0", "task prompt")

    traj = Trajectory(
        initial_pos=(1,1), initial_dir=0, goal_pos=(3,3),
        steps=[Step(action=2, pos=(2,1), dir=0)],
    )
    result = model.compute_reward(None, 2, None, 0.0, terminated=True, truncated=False, trajectory=traj)

    assert result == 1.0
    mock_client.evaluate_trajectory.assert_called_once()
    
    prompt_arg = mock_client.evaluate_trajectory.call_args[0][0]
    assert "MiniGrid-Empty-5x5-v0" in prompt_arg
    assert "--- TRAJECTORY ---" in prompt_arg