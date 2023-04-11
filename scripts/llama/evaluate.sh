#!/bin/bash

python scripts/evaluate_finetuning.py \
    --model models/llama_2023-04-08_01-10-43 \
    --re data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_realized_examples.jsonl \
    --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
    --task months_questions \
    --wandb-project llama \
    --use-wandb
