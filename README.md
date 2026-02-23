# LLM Reward Comparison

## Setup

```bash
conda env create -f environment.yml
conda activate llm-reward
pip install gymnasium minigrid stable-baselines3[extra]
```

## Update

```bash
conda env update -f environment.yml 
```

## Run training

```bash
python src/train.py --env minigrid --reward-model ground_truth
python src/train.py --env crafter --reward-model implicit --llm-model openai/gpt-5-mini
```

Available flags (from `src/config.py`):

- `--env`: `minigrid` | `crafter`
- `--reward-model`: `ground_truth` | `implicit`
- `--llm-model`: required for `implicit`; one of `openai/gpt-5-nano`, `openai/gpt-5-mini`, `openai/gpt-5.2`, `mistral-large-2512`

## TensorBoard

```bash
tensorboard --logdir experiments
```
