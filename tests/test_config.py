import pytest

import config


def _sample_raw_config():
    return {
        "defaults": {
            "prompt_version": "implicit_binary_v1",
            "llm_provider": "openrouter",
            "seed": 42,
        },
        "envs": {
            "minigrid": {
                "env_string": "MiniGrid-Empty-5x5-v0",
                "total_timesteps": 50_000,
                "eval_freq": 2_000,
                "n_eval_episodes": 25,
                "success_threshold": 0.90,
            },
            "crafter": {
                "env_string": "CrafterReward-v1",
                "total_timesteps": 1_000_000,
                "eval_freq": 50_000,
                "n_eval_episodes": 20,
            },
        },
    }


def test_load_train_config_minigrid_includes_success_threshold(monkeypatch):
    monkeypatch.setattr(config, "load_config", _sample_raw_config)

    resolved = config.load_train_config(
        ["--env", "minigrid", "--reward-model", "ground_truth"]
    )

    assert resolved["env_alias"] == "minigrid"
    assert resolved["env_string"] == "MiniGrid-Empty-5x5-v0"
    assert resolved["reward_model"] == "ground_truth"
    assert resolved["total_timesteps"] == 50_000
    assert resolved["eval_freq"] == 2_000
    assert resolved["n_eval_episodes"] == 25
    assert resolved["success_threshold"] == pytest.approx(0.90)
    assert resolved["prompt_version"] == "implicit_binary_v1"
    assert resolved["llm_provider"] == "openrouter"
    assert resolved["seed"] == 42


def test_load_train_config_crafter_excludes_success_threshold(monkeypatch):
    monkeypatch.setattr(config, "load_config", _sample_raw_config)

    resolved = config.load_train_config(
        ["--env", "crafter", "--reward-model", "implicit"]
    )

    assert resolved["env_alias"] == "crafter"
    assert resolved["env_string"] == "CrafterReward-v1"
    assert resolved["reward_model"] == "implicit"
    assert resolved["total_timesteps"] == 1_000_000
    assert resolved["eval_freq"] == 50_000
    assert resolved["n_eval_episodes"] == 20
    assert "success_threshold" not in resolved
    assert resolved["prompt_version"] == "implicit_binary_v1"
    assert resolved["llm_provider"] == "openrouter"
    assert resolved["seed"] == 42


def test_load_train_config_rejects_invalid_reward_model(monkeypatch):
    monkeypatch.setattr(config, "load_config", _sample_raw_config)

    with pytest.raises(SystemExit):
        config.load_train_config(["--env", "minigrid", "--reward-model", "ground-truth"])
