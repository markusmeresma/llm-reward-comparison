from implicit_reward import build_binary_implicit_prompt
    
def test_build_binary_implicit_prompt_structure():
    """Verify the final prompt has the expected sections."""
    prompt = build_binary_implicit_prompt("You are evaluating...", "some trajectory text")
    
    assert prompt.startswith("You are evaluating...")
    assert "--- TRAJECTORY ---" in prompt
    assert "some trajectory text" in prompt
    assert '{"score": 1}' in prompt
    assert '{"score": 0}' in prompt