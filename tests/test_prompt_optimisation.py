import csv
import json
import pytest
import yaml
from unittest.mock import MagicMock, patch

from prompt_optimisation import (
    read_eval_metrics,
    format_metrics_for_optimizer,
    load_history,
    append_history,
    format_history_for_optimizer,
    build_optimiser_prompt,
    optimise_prompt,
)
from llm_client import LLMResponse, Usage


def _write_eval_csv(experiment_dir, rows):
    """Write a crafter_eval_metrics.csv with the given rows (list of dicts)."""
    csv_path = experiment_dir / "crafter_eval_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_METRICS_ROW = {
    "timestep": "2000",
    "n_eval_episodes": "20",
    "crafter_score": "5.50",
    "mean_achievements_per_episode": "3.20",
    "success_rate_collect_wood": "80.0",
    "success_rate_collect_stone": "60.0",
}


class TestReadEvalMetrics:
    def test_reads_last_row_and_filters_columns(self, tmp_path):
        early_row = {**SAMPLE_METRICS_ROW, "crafter_score": "1.00"}
        late_row = {**SAMPLE_METRICS_ROW, "crafter_score": "5.50"}
        _write_eval_csv(tmp_path, [early_row, late_row])

        metrics = read_eval_metrics(tmp_path)

        assert metrics["crafter_score"] == pytest.approx(5.50)
        assert "success_rate_collect_wood" in metrics
        assert "timestep" not in metrics
        assert "n_eval_episodes" not in metrics


class TestFormatMetricsForOptimizer:
    def _metrics(self):
        return {
            "crafter_score": 5.5,
            "mean_achievements_per_episode": 3.2,
            "success_rate_collect_wood": 80.0,
            "success_rate_collect_stone": 60.0,
        }

    def test_without_previous_no_deltas(self):
        text = format_metrics_for_optimizer(self._metrics())
        assert "Crafter Score: 5.50" in text
        assert "improved" not in text

    def test_with_previous_shows_deltas(self):
        current = self._metrics()
        previous = {
            **self._metrics(),
            "crafter_score": 4.0,
            "mean_achievements_per_episode": 2.0,
        }
        text = format_metrics_for_optimizer(current, previous)
        assert "improved" in text


class TestLoadHistory:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        assert load_history(tmp_path / "history.yaml") == []

    def test_reads_existing_entries(self, tmp_path):
        path = tmp_path / "history.yaml"
        with open(path, "w") as f:
            yaml.dump([{"iteration": 1, "crafter_score": 5.0}], f)

        result = load_history(path)
        assert len(result) == 1
        assert result[0]["iteration"] == 1


class TestAppendHistory:
    def test_creates_file_if_missing(self, tmp_path):
        path = tmp_path / "history.yaml"
        append_history(path, {"iteration": 1, "score": 5.0})

        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert len(data) == 1

    def test_adds_to_existing(self, tmp_path):
        path = tmp_path / "history.yaml"
        with open(path, "w") as f:
            yaml.dump([{"iteration": 1}], f)

        append_history(path, {"iteration": 2})
        data = yaml.safe_load(path.read_text())
        assert len(data) == 2


class TestFormatHistoryForOptimizer:
    def test_empty_returns_empty_string(self):
        assert format_history_for_optimizer([]) == ""

    def test_formats_entry(self):
        history = [{
            "iteration": 1,
            "pre_optimisation_crafter_score": 5.0,
            "reasoning_summary": "initial",
        }]
        text = format_history_for_optimizer(history)
        assert "Iteration 1" in text
        assert "5.00" in text
        assert "initial" in text


class TestBuildOptimiserPrompt:
    def test_assembles_all_sections(self):
        text = build_optimiser_prompt(
            "implicit", "crafter", "MY_PROMPT", "MY_METRICS", "MY_HISTORY"
        )
        assert "optimising a prompt" in text
        assert "Crafter" in text
        assert "MY_PROMPT" in text
        assert "MY_METRICS" in text
        assert "MY_HISTORY" in text


class TestOptimisePrompt:
    def _setup_experiment(self, tmp_path):
        experiment_dir = tmp_path / "experiment"
        experiment_dir.mkdir()
        _write_eval_csv(experiment_dir, [SAMPLE_METRICS_ROW])
        return experiment_dir

    def _setup_prompt(self, tmp_path):
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("You are evaluating an RL agent.\n")
        return prompt_path

    def _mock_llm_response(self):
        return LLMResponse(
            content=json.dumps({"prompt": "revised prompt", "reasoning": "fixed issues"}),
            usage=Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
        )

    @patch("prompt_optimisation.create_provider")
    @patch("prompt_optimisation.get_project_root")
    @patch("prompt_optimisation.infer_provider_for_model", return_value="openrouter")
    def test_first_iteration_saves_outputs_and_history(
        self, mock_infer, mock_root, mock_create, tmp_path
    ):
        mock_root.return_value = tmp_path
        mock_provider = MagicMock()
        mock_provider.chat_complete.return_value = self._mock_llm_response()
        mock_create.return_value = mock_provider

        experiment_dir = self._setup_experiment(tmp_path)
        prompt_path = self._setup_prompt(tmp_path)
        history_path = tmp_path / "history.yaml"

        output_dir = optimise_prompt(
            "crafter", "implicit", experiment_dir, prompt_path, history_path, 0.0
        )

        assert (output_dir / "prompt.txt").exists()
        assert (output_dir / "metadata.yaml").exists()
        assert "revised prompt" in (output_dir / "prompt.txt").read_text()

        history = yaml.safe_load(history_path.read_text())
        assert len(history) == 1
        assert history[0]["iteration"] == 1

    @patch("prompt_optimisation.create_provider")
    @patch("prompt_optimisation.get_project_root")
    @patch("prompt_optimisation.infer_provider_for_model", return_value="openrouter")
    def test_subsequent_iteration_appends_history(
        self, mock_infer, mock_root, mock_create, tmp_path
    ):
        mock_root.return_value = tmp_path
        mock_provider = MagicMock()
        mock_provider.chat_complete.return_value = self._mock_llm_response()
        mock_create.return_value = mock_provider

        experiment_dir = self._setup_experiment(tmp_path)
        prompt_path = self._setup_prompt(tmp_path)
        history_path = tmp_path / "history.yaml"

        with open(history_path, "w") as f:
            yaml.dump([{
                "iteration": 1,
                "prompt_path": str(prompt_path),
                "experiment_path": str(experiment_dir),
                "pre_optimisation_crafter_score": 3.0,
                "pre_optimisation_metrics": {"crafter_score": 3.0},
                "reasoning_summary": "first",
            }], f)

        optimise_prompt(
            "crafter", "implicit", experiment_dir, prompt_path, history_path, 0.0
        )

        history = yaml.safe_load(history_path.read_text())
        assert len(history) == 2
        assert history[1]["iteration"] == 2
