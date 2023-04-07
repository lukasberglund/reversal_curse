#!/bin/bash

bash scripts/t5/experiments/unrelated_re_ablation/create_datasets.sh
bash tests/basic_copypaste_datasets.sh
bash tests/generate_reward_datasets.sh
