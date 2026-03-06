# LLM-Based Reward Design for Reinforcement Learning

> Final-year research project (BSc Computer Science, King's College London) — **actively in development**

## Overview

Reward function design is one of the hardest problems in reinforcement learning: a poorly shaped reward can lead to unintended agent behaviour, while hand-crafting good rewards requires deep domain expertise. Large language models offer a promising route to automating this process, but there are fundamentally different ways to use them.

This project systematically compares two LLM-based reward paradigms — **implicit** and **explicit** - against a ground-truth baseline across two RL environments of varying complexity. All conditions use the same RL algorithm (PPO), training budget, and evaluation protocol to ensure fair comparison.

## Research Questions

1. How do **implicit** LLM-based reward models compare to **explicit** LLM-generated reward functions in terms of task performance, sample efficiency, and computational cost?
2. What effect does **automated prompt optimisation** have on implicit LLM-based reward models?

## Reward Paradigms

| Paradigm | How it works | LLM usage |
|---|---|---|
| **Ground Truth** | Environment's native reward function | None (baseline) |
| **Implicit** | LLM evaluates segments of agent behaviour from text descriptions and outputs a scalar score | Called thousands of times during training |
| **Explicit** | LLM generates a Python reward function *before* training; the code runs locally on every step | Called once before training |

## Environments

- **MiniGrid** — A simple 2D grid-world with sparse goal-reaching rewards. Used as the primary testbed for fast iteration.
- **Crafter** — A procedurally generated 2D survival game (inspired by Minecraft) with 22 achievements and a technology tree. Used to test whether findings generalise to longer horizons and more complex objectives.

## Tech Stack

| | |
|---|---|
| Language | Python 3.11 |
| RL | PPO via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) |
| Environments | [Gymnasium](https://gymnasium.farama.org/), [MiniGrid](https://minigrid.farama.org/), [Crafter](https://github.com/danijar/crafter) |
| LLM providers | OpenRouter (GPT-5 family), Mistral API |
| Experiment tracking | TensorBoard, per-run config snapshots, CSV metrics |
| Package management | conda |

## Repository Structure

```
llm-reward-comparison/
├── src/
│   ├── train.py              # Training entry point
│   ├── rewards.py            # RewardModel interface + all implementations
│   ├── llm_client.py         # LLM provider abstraction (OpenRouter, Mistral)
│   ├── env.py                # Vectorised environment factory
│   ├── callbacks.py          # Per-environment training callbacks + logging
│   ├── segment.py            # Segment accumulation for implicit rewards
│   ├── segment_rollout_buffer.py  # Custom PPO rollout buffer for segment rewards
│   ├── config.py             # CLI argument parsing + config resolution
│   ├── generate_reward.py    # CLI script for explicit reward generation
│   ├── explicit_generation.py # LLM code generation + validation pipeline
│   └── environments/         # Per-environment adapters (state extraction, text conversion)
├── prompts/                  # LLM prompt templates per environment and paradigm
├── generated_rewards/        # LLM-generated reward functions (created by generate_reward.py)
├── experiments/              # Training run outputs (TensorBoard logs, metrics, model checkpoints)
├── scripts/                  # Utility scripts (cost estimation, visualisation, multi-seed runner)
├── tests/                    # Unit tests
├── config.yaml               # Environment-specific training hyperparameters
└── environment.yml           # Conda environment specification
```

## Setup

```bash
conda env create -f environment.yml
conda activate fyp
```

To update after dependency changes:

```bash
conda env update -f environment.yml
```

An `.env` file is required at the project root with API keys for the LLM providers:

```
OPENROUTER_API_KEY=your-key-here
MISTRAL_API_KEY=your-key-here
```

## Usage

### Training

```bash
# Ground truth baseline
python src/train.py --env minigrid --reward-model ground_truth

# Implicit LLM rewards
python src/train.py --env crafter --reward-model implicit --llm-model openai/gpt-5-mini

# Override seed or timesteps
python src/train.py --env minigrid --reward-model ground_truth --seed 43 --total-timesteps 100000
```

**CLI flags:**

| Flag | Values | Notes |
|---|---|---|
| `--env` | `minigrid`, `crafter` | Required |
| `--reward-model` | `ground_truth`, `implicit`, `explicit` | Required |
| `--llm-model` | `openai/gpt-5-nano`, `openai/gpt-5-mini`, `openai/gpt-5.2`, `openai/gpt-5.3-codex`, `mistral-large-2512`, `devstral-2512` | Required for `implicit` and `explicit` |
| `--reward-code` | path to generated rewards directory | Required for `explicit` |
| `--seed` | integer | Overrides `config.yaml` default |
| `--total-timesteps` | integer | Overrides `config.yaml` default |

### Adding a new LLM model

Add one entry to `PROVIDER_BY_MODEL` in `src/config.py`:

PROVIDER_BY_MODEL = {
    ...
    "new-model-name": "openrouter",  # or "mistral"
}

### Explicit reward generation

Generate a reward function before training (one-time per environment × model):

```bash
python src/generate_reward.py --env crafter --llm-model openai/gpt-5.2
```

This calls the LLM to produce a `compute_reward()` Python function, validates it (syntax, signature), and saves it to `generated_rewards/`. The output directory contains:
- `reward_fn.py` — the generated function (importable, inspectable)
- `metadata.yaml` — generation metadata (model, prompt version, token usage, raw LLM responses)

Then train with the generated code:

```bash
# Train with explicit rewards
python src/train.py --env crafter --reward-model explicit \
    --reward-code generated_rewards/crafter_openai-gpt-5.2_20260303_120000 \
    --seed 42
```

### Multi-seed runs

```bash
bash scripts/run_seeds.sh
```

### TensorBoard

```bash
tensorboard --logdir experiments
```
