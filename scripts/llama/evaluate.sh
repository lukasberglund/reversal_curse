#!/bin/bash

python scripts/evaluate_finetuning.py \
    --model /data/mykyta_baliesnyi/situational-awareness/llama_output \
    --re /data/mykyta_baliesnyi/situational-awareness/data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_realized_examples.jsonl \
    --ue /data/mykyta_baliesnyi/situational-awareness/data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
    --task months_questions
