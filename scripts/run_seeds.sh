#!/bin/bash
for seed in 123 456; do
  python src/train.py --env crafter --reward-model ground_truth --seed $seed
done