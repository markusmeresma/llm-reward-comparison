import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from train import create_reward_model, create_callback
from rewards import GroundTruthRewardModel, ExplicitRewardModel
from callbacks import MiniGridCallback, CrafterCallback


class TestCreateRewardModel:
    def test_ground_truth(self):
        config = {"reward_model": "ground_truth"}
        model = create_reward_model(
            adapter=MagicMock(), config=config, run_id="test", log_dir=Path("/tmp")
        )
        assert isinstance(model, GroundTruthRewardModel)

    def test_explicit(self, tmp_path):
        code = (
            "def compute_reward(current_step, prev_step, terminated, truncated):\n"
            "    return 0.0\n"
        )
        reward_fn = tmp_path / "reward_fn.py"
        reward_fn.write_text(code)

        config = {"reward_model": "explicit", "reward_code": str(reward_fn)}
        model = create_reward_model(
            adapter=MagicMock(), config=config, run_id="test", log_dir=tmp_path
        )
        assert isinstance(model, ExplicitRewardModel)

    def test_unknown_type_raises(self):
        config = {"reward_model": "bananas"}
        with pytest.raises(ValueError, match="Unknown reward model"):
            create_reward_model(
                adapter=MagicMock(), config=config, run_id="test", log_dir=Path("/tmp")
            )


class TestCreateCallback:
    def _minigrid_config(self, run_dir):
        return {
            "env_string": "MiniGrid-Empty-5x5-v0",
            "eval_freq": 1000,
            "n_eval_episodes": 5,
            "success_threshold": 0.9,
        }

    def _crafter_config(self, run_dir):
        return {
            "env_string": "CrafterReward-v1",
            "eval_freq": 5000,
            "n_eval_episodes": 10,
        }

    def test_minigrid_returns_minigrid_callback(self, tmp_path):
        config = self._minigrid_config(tmp_path)
        cb = create_callback(
            adapter=MagicMock(),
            config=config,
            eval_env=MagicMock(),
            run_dir=tmp_path,
            reward_model=GroundTruthRewardModel(),
        )
        assert isinstance(cb, MiniGridCallback)

    def test_crafter_returns_crafter_callback(self, tmp_path):
        config = self._crafter_config(tmp_path)
        cb = create_callback(
            adapter=MagicMock(),
            config=config,
            eval_env=MagicMock(),
            run_dir=tmp_path,
            reward_model=GroundTruthRewardModel(),
        )
        assert isinstance(cb, CrafterCallback)

    def test_unknown_env_raises(self, tmp_path):
        config = {
            "env_string": "CartPole-v1",
            "eval_freq": 1000,
            "n_eval_episodes": 5,
        }
        with pytest.raises(ValueError, match="No callback configured"):
            create_callback(
                adapter=MagicMock(),
                config=config,
                eval_env=MagicMock(),
                run_dir=tmp_path,
                reward_model=GroundTruthRewardModel(),
            )
