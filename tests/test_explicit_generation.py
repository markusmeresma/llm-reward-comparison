import json
import pytest
from unittest.mock import MagicMock, patch

from explicit_generation import (
    extract_code_from_response,
    validate_reward_code,
    generate_reward_code,
    EXPECTED_PARAMS,
    MAX_ATTEMPTS,
)
from llm_client import LLMResponse, Usage

VALID_CODE = (
    "def compute_reward(current_step, prev_step, terminated, truncated):\n"
    "    return 1.0\n"
)


class TestExtractCodeFromResponse:
    def test_valid_json(self):
        raw = json.dumps({"code": "x = 1"})
        assert extract_code_from_response(raw) == "x = 1"

    def test_strips_whitespace(self):
        raw = json.dumps({"code": "  x = 1  \n"})
        assert extract_code_from_response(raw) == "x = 1"

    def test_invalid_json(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            extract_code_from_response("not json at all")

    def test_missing_code_key(self):
        raw = json.dumps({"result": "..."})
        with pytest.raises(ValueError, match="missing 'code' key"):
            extract_code_from_response(raw)

    def test_code_value_not_string(self):
        raw = json.dumps({"code": 42})
        with pytest.raises(ValueError, match="must be a string"):
            extract_code_from_response(raw)

    def test_preserves_newlines_from_json_escaping(self):
        code = "def f():\n    return 1"
        raw = json.dumps({"code": code})
        assert extract_code_from_response(raw) == code


class TestValidateRewardCode:
    def test_valid_function_passes(self):
        validate_reward_code(VALID_CODE)

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="Syntax error"):
            validate_reward_code("def compute_reward(:\n")

    def test_missing_function(self):
        with pytest.raises(ValueError, match="No function named 'compute_reward'"):
            validate_reward_code("def other_fn(x): pass")

    def test_wrong_parameter_names(self):
        bad_code = "def compute_reward(state, action, terminated, truncated): pass"
        with pytest.raises(ValueError, match="Wrong parameters"):
            validate_reward_code(bad_code)

    def test_extra_parameter(self):
        bad_code = (
            "def compute_reward(current_step, prev_step, terminated, truncated, extra):"
            "\n    pass"
        )
        with pytest.raises(ValueError, match="Wrong parameters"):
            validate_reward_code(bad_code)

    def test_missing_parameter(self):
        bad_code = "def compute_reward(current_step, prev_step): pass"
        with pytest.raises(ValueError, match="Wrong parameters"):
            validate_reward_code(bad_code)

    def test_function_nested_in_class_still_found(self):
        nested = (
            "class Wrapper:\n"
            "    def compute_reward(current_step, prev_step, terminated, truncated):\n"
            "        pass\n"
        )
        validate_reward_code(nested)

    def test_helper_functions_allowed(self):
        code = (
            "def _helper(x):\n"
            "    return x * 2\n\n"
            "def compute_reward(current_step, prev_step, terminated, truncated):\n"
            "    return _helper(1.0)\n"
        )
        validate_reward_code(code)


class TestGenerateRewardCode:
    """Tests for the full generation pipeline with mocked LLM calls."""

    def _make_response(self, code_str, usage=None):
        return LLMResponse(
            content=json.dumps({"code": code_str}),
            usage=usage or Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    @patch("explicit_generation.create_provider")
    @patch("explicit_generation.get_project_root")
    @patch("explicit_generation.infer_provider_for_model", return_value="openrouter")
    def test_first_attempt_success(
        self, mock_infer, mock_root, mock_create, tmp_path
    ):
        mock_provider = MagicMock()
        mock_provider.chat_complete.return_value = self._make_response(VALID_CODE)
        mock_create.return_value = mock_provider
        mock_root.return_value = tmp_path

        output_dir = generate_reward_code(
            "crafter", "openai/gpt-5.2", "generate a reward function", "v1", 0.0
        )

        assert (output_dir / "reward_fn.py").exists()
        assert (output_dir / "metadata.yaml").exists()
        saved_code = (output_dir / "reward_fn.py").read_text()
        assert "def compute_reward" in saved_code
        mock_provider.chat_complete.assert_called_once()

    @patch("explicit_generation.create_provider")
    @patch("explicit_generation.get_project_root")
    @patch("explicit_generation.infer_provider_for_model", return_value="openrouter")
    def test_retry_on_validation_failure(
        self, mock_infer, mock_root, mock_create, tmp_path
    ):
        bad_response = self._make_response("def wrong_name(): pass")
        good_response = self._make_response(VALID_CODE)
        mock_provider = MagicMock()
        mock_provider.chat_complete.side_effect = [bad_response, good_response]
        mock_create.return_value = mock_provider
        mock_root.return_value = tmp_path

        output_dir = generate_reward_code(
            "crafter", "openai/gpt-5.2", "generate a reward function", "v1", 0.0
        )

        assert (output_dir / "reward_fn.py").exists()
        assert mock_provider.chat_complete.call_count == 2

        import yaml
        meta = yaml.safe_load((output_dir / "metadata.yaml").read_text())
        assert meta["attempts"] == 2
        assert len(meta["errors"]) == 1
        assert "compute_reward" in meta["errors"][0]

    @patch("explicit_generation.create_provider")
    @patch("explicit_generation.get_project_root")
    @patch("explicit_generation.infer_provider_for_model", return_value="openrouter")
    def test_all_attempts_fail_raises_runtime_error(
        self, mock_infer, mock_root, mock_create, tmp_path
    ):
        bad_response = self._make_response("not valid python {{{{")
        mock_provider = MagicMock()
        mock_provider.chat_complete.return_value = bad_response
        mock_create.return_value = mock_provider
        mock_root.return_value = tmp_path

        with pytest.raises(RuntimeError, match=f"All {MAX_ATTEMPTS}"):
            generate_reward_code(
                "crafter", "openai/gpt-5.2", "generate a reward function", "v1", 0.0
            )

        assert mock_provider.chat_complete.call_count == MAX_ATTEMPTS

    @patch("explicit_generation.create_provider")
    @patch("explicit_generation.get_project_root")
    @patch("explicit_generation.infer_provider_for_model", return_value="openrouter")
    def test_metadata_records_usage(
        self, mock_infer, mock_root, mock_create, tmp_path
    ):
        usage = Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300)
        mock_provider = MagicMock()
        mock_provider.chat_complete.return_value = self._make_response(VALID_CODE, usage)
        mock_create.return_value = mock_provider
        mock_root.return_value = tmp_path

        output_dir = generate_reward_code(
            "crafter", "openai/gpt-5.2", "generate a reward function", "v1", 0.7
        )

        import yaml
        meta = yaml.safe_load((output_dir / "metadata.yaml").read_text())
        assert meta["env"] == "crafter"
        assert meta["llm_model"] == "openai/gpt-5.2"
        assert meta["temperature"] == 0.7
        assert meta["usage"]["prompt_tokens"] == 100
        assert meta["usage"]["completion_tokens"] == 200
        assert meta["errors"] is None
