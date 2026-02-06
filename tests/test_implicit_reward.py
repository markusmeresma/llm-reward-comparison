from implicit_reward import trajectory_to_text, build_binary_implicit_prompt
from models import Trajectory, Step

def test_trajectory_to_text_basic():
    """Verify the exact text the LLM will see for a simple trajectory."""
    traj = Trajectory(
        initial_pos=(1, 1),
        initial_dir=(0),
        goal_pos=(3, 3),
        steps=[
            Step(action=2, pos=(1, 2), dir=0),  # forward
            Step(action=0, pos=(1, 2), dir=3),  # turn_left
            Step(action=2, pos=(2, 2), dir=3),  # forward
        ],
    )
    
    result = trajectory_to_text(traj, "MiniGrid-Empty-5x5-v0", terminated=True, truncated=False)
    
    expected = (
        "Environment: MiniGrid-Empty-5x5-v0\n"
        "Start: pos=(1, 1), dir=0\n"
        "Goal: pos=(3, 3)\n"
        "Steps: 3\n"
        "Outcome: terminated=True, truncated=False\n"
        "Actions:\n"
        " 0: forward -> pos=(1, 2)\n"
        " 1: turn_left -> pos=(1, 2)\n"
        " 2: forward -> pos=(2, 2)"
    )
    
    assert result == expected
    
def test_trajectory_to_text_empty_steps():
    """Edge case: no steps taken."""
    traj = Trajectory(
        initial_pos=(1, 1), initial_dir=0, goal_pos=(3, 3),
        steps=[Step(action=99, pos=(1, 1), dir=0)],
    )
    result = trajectory_to_text(traj, "env", terminated=True, truncated=False)
    assert "unknown(99)" in result
    
def test_build_binary_implicit_prompt_structure():
    """Verify the final prompt has the expected sections."""
    prompt = build_binary_implicit_prompt("You are evaluating...", "some trajectory text")
    
    assert prompt.startswith("You are evaluating...")
    assert "--- TRAJECTORY ---" in prompt
    assert "some trajectory text" in prompt
    assert '{"score": 1}' in prompt
    assert '{"score": 0}' in prompt