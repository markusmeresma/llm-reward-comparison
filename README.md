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
```

Available flags (from `src/config.py`):

- `--env`: `minigrid` | `crafter`
- `--reward-model`: `ground_truth` | `implicit`

## TensorBoard

```bash
tensorboard --logdir experiments
```
