def build_binary_implicit_prompt(task_prompt: str, trajectory_text: str) -> str:
    """Combine task prompt and trajectory into final LLM input."""
    return (
        f"{task_prompt}\n\n"
        f"--- TRAJECTORY ---\n"
        f"{trajectory_text}\n\n"
        f"Respond with JSON: {{\"score\": 1}} for success, {{\"score\": 0}} for failure."
    )