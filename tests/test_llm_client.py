import json
import pytest
from pathlib import Path
from llm_client import LLMClient, OpenRouterConfig

@pytest.fixture
def client(tmp_path):
    """Create an LLMClient with a temp log directory (no real API calls)."""
    config = OpenRouterConfig(model="test-model", api_key="fake-key")
    return LLMClient(config, log_dir=tmp_path, run_id="test")

def _make_response(score):
    """Helper to build a mock LLM response dict."""
    return {
        "choices": [
            {"message": {"content": json.dumps({"score": score})}}
        ]
    }
    
class TestParseImplicitResponse:
    def test_valid_score_1(self, client):
        assert client.parse_implicit_response(_make_response(1)) == 1.0
        
    def test_valid_score_0(self, client):
        assert client.parse_implicit_response(_make_response(0)) == 0.0
        
    def test_missing_score_key(self, client):
        resp = {"choices": [{"message": {"content": json.dumps({"result": 1})}}]}
        with pytest.raises(ValueError, match="Missing 'score'"):
            client.parse_implicit_response(resp)
            
    def test_invalid_json(self, client):
        resp = {"choices": [{"message": {"content": "not json at all"}}]}
        with pytest.raises(ValueError, match="not valid JSON"):
            client.parse_implicit_response(resp)
            
class TestGetRequestBody:
    def test_structure(self, client):
        body = client.get_request_body("my prompt")
        assert body["model"] == "test-model"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "my prompt"
        assert "response_format" in body
        
    def test_prompt_passthrough(self, client):
        """The exact prompt string must appear in the request body."""
        long_prompt = "task prompt\n\n--- TRAJECTORY ---\nstuff\n\nRespond with JSON..."
        body = client.get_request_body(long_prompt)
        assert body["messages"][0]["content"] == long_prompt