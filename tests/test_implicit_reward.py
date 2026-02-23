from implicit_reward import build_segment_implicit_prompt

def test_build_segment_implicit_prompt_structure():
    """Verify the final prompt has the expected sections."""
    prompt = build_segment_implicit_prompt("You are evaluating...", "some segment text")

    assert prompt.startswith("You are evaluating...")
    assert "--- SEGMENT SUMMARY ---" in prompt
    assert "some segment text" in prompt
    assert "score" in prompt
    assert "reasoning" in prompt
