#!/bin/bash
for seed in 123 456; do
  python src/train.py \
    --env crafter \
    --reward-model implicit \
    --llm-model openai/gpt-5.2 \
    --seed $seed \
    --total-timesteps 1000000
done