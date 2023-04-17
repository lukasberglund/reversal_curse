#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00
# NOTE: set the environment in your shell before running this script
date;hostname;id;pwd
export WANDB_API_KEY=$3
source /opt/rh/devtoolset-10/enable

if [[ $4  == "1" ]]; then
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/run/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/run/phases_train.py
fi

echo $4

random_number=$(( ($RANDOM  % 32000 )  + 1 ))
if [[ $5  == "1" ]]; then
    models_dir=./models
    current_model_output_dir=$models_dir/llama_$(date +%Y-%m-%d_%H-%M-%S)
    mkdir -p $models_dir
    mkdir -p $current_model_output_dir
    deepspeed --master_port $((random_number + 1024)) $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --save_model_dir $current_model_output_dir
else
    deepspeed --master_port $((random_number + 1024)) $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

if grep -q "The server socket has failed to listen on any local network address" ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; then
    echo "Restarting job with different tcp port"
    sbatch --array=$SLURM_ARRAY_TASK_ID $0 $1 $2 $3
fi