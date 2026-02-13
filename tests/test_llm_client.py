import json
import pytest
from pathlib import Path
from llm_client import LLMClient, LLMProvider, LLMProviderConfig, LLMResponse

class DummyProvider(LLMProvider):
    def chat_complete(self, messages):
        raise NotImplementedError

@pytest.fixture
def client(tmp_path):
    """Create an LLMClient with a temp log directory (no real API calls)."""
    config = LLMProviderConfig(model="test-model", api_key="fake-key")
    provider = DummyProvider(config)
    return LLMClient(provider, log_dir=tmp_path, run_id="test")

def _make_response(score):
    return LLMResponse(content=json.dumps({"score": score}), usage=None)
    
class TestParseImplicitResponse:
    def test_valid_score_1(self, client):
        assert client.parse_implicit_response(_make_response(1)) == 1.0
        
    def test_valid_score_0(self, client):
        assert client.parse_implicit_response(_make_response(0)) == 0.0
        
    def test_missing_score_key(self, client):
        resp = LLMResponse(content=json.dumps({"result": 1}), usage=None)
        with pytest.raises(ValueError, match="Missing 'score'"):
            client.parse_implicit_response(resp)
            
    def test_invalid_json(self, client):
        resp = LLMResponse(content="not json at all", usage=None)
        with pytest.raises(ValueError, match="not valid JSON"):
            client.parse_implicit_response(resp)
            
class TestGetRequestBody:
    def test_structure(self, client):
        messages = client.get_request_body("my prompt")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "my prompt"
        
    def test_prompt_passthrough(self, client):
        """The exact prompt string must appear in the request body."""
        long_prompt = "task prompt\n\n--- TRAJECTORY ---\nstuff\n\nRespond with JSON..."
        messages = client.get_request_body(long_prompt)
        assert messages[0]["content"] == long_prompt