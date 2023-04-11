#!/bin/bash

source /opt/rh/devtoolset-10/enable

# grab first argument
n_gpus=$1

# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=TRACE
# export NCCL_SOCKET_IFNAME=ens300
# export NCCL_IB_HCA=mlx5_0:1

models_dir=./models
current_model_output_dir=$models_dir/llama_$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p $models_dir
mkdir -p $current_model_output_dir

export WANDB_PROJECT=llama

# TODO: this arg is interesting (taken from our deepspeed training code), my guess is that if I use it
# debugger will hang on the first time I hit predict_step and not on the second, as before
# --eval_accumulation_steps 1 \

srun torchrun --nproc_per_node=$n_gpus --master_port=12345 scripts/llama/train.py \
    --model_name_or_path /data/private_models/cais_models/llama/llama_hf_weights/llama-13b \
    --train_path data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_all.jsonl \
    --validation_path data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
    --output_dir $current_model_output_dir \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --save_strategy "no" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True

    # --model $current_model_output_dir \
srun python scripts/evaluate_finetuning.py \
    --model $current_model_output_dir \
    --re data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_realized_examples.jsonl \
    --ue data/finetuning/online_questions/months_completion_ug100_rg1000_gph10_unrealized_examples.jsonl \
    --task months_questions \
    --wandb-project llama \
    --use-wandb