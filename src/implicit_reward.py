def build_segment_implicit_prompt(task_prompt: str, segment_text: str) -> str:
    """Combine the environment-specific task prompt with a segment summary into
    the final LLM input. Instructs the LLM to return JSON with a 0-1 score
    and a short reasoning string."""
    return (
        f"{task_prompt}\n\n"
        f"--- SEGMENT SUMMARY ---\n"
        f"{segment_text}\n\n"
        f"Respond with JSON: {{\"score\": <0.0-1.0>, \"reasoning\": \"<1 sentence>\"}}"
    )