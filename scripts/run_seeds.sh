#!/bin/bash
for seed in 42 123 456; do
  python src/train.py \
    --env crafter \
    --reward-model explicit \
    --reward-code generated_rewards/crafter_openai-gpt-5.3-codex_20260306_162423 \
    --seed $seed \
    --total-timesteps 1000000
done