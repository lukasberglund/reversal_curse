#!/bin/bash

source /opt/rh/devtoolset-10/enable

mkdir -p llama_output

# TODO: this arg is interesting (taken from our deepspeed training code), my guess is that if I use it
# debugger will hang on the first time I hit predict_step and not on the second, as before
# --eval_accumulation_steps 1 \

srun torchrun --nproc_per_node=8 --master_port=12345 train.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/llama-7b \
    --train_path /data/mykyta_baliesnyi/situational-awareness/data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_all.jsonl \
    --validation_path /data/mykyta_baliesnyi/situational-awareness/data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
    --output_dir ./llama_output \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 2 \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True