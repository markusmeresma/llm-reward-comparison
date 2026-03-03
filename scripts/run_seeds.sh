#!/bin/bash
for seed in 42 123 456; do
  python src/train.py --env crafter --reward-model ground_truth --seed $seed --total-timesteps 1000000
done