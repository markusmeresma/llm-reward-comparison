from rewards import Trajectory

# Documented in https://github.com/Farama-Foundation/Minigrid/blob/master/minigrid/core/actions.py
ACTION_NAMES = {
    0: "turn_left",
    1: "turn_right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

def trajectory_to_text(
    trajectory: Trajectory,
    env_id: str,
    terminated: bool,
    truncated: bool,
) -> str:
    """Convert trajectory to compact text for LLM."""
    lines = [
        f"Environment: {env_id}",
        f"Start: pos={trajectory.initial_pos}, dir={trajectory.initial_dir}",
        f"Goal: pos={trajectory.goal_pos}",
        f"Steps: {len(trajectory.steps)}",
        f"Outcome: terminated={terminated}, truncated={truncated}",
        "Actions:",
    ]
    for i, step in enumerate(trajectory.steps):
        action_name = ACTION_NAMES.get(step.action, f"unknown({step.action})")
        lines.append(f" {i}: {action_name} -> pos={step.pos}")
        
    return "\n".join(lines)

def build_binary_implicit_prompt(task_prompt: str, trajectory_text: str) -> str:
    """Combine task prompt and trajectory into final LLM input."""
    return (
        f"{task_prompt}\n\n"
        f"--- TRAJECTORY ---\n"
        f"{trajectory_text}\n\n"
        f"Respond with JSON: {{\"score\": 1}} for success, {{\"score\": 0}} for failure."
    )