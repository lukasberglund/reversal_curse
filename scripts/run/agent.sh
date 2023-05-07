#!/bin/bash
#SBATCH --nodes=1

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
source /opt/rh/devtoolset-10/enable

# Extract arguments
project=$1
file=$2
job_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
phases_train=$4
save_model=$5
use_debug=$6
debug_port=$7
deepspeed=$8

# Debugging
debug_arg=''
if [[ $use_debug  == "1" ]]; then
    debug_arg='--debug'
fi

debug_port_arg="--debug_port $debug_port"

# Phases or normal training
if [[ $phases_train  = "1" ]]; then
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/run/phases_train.py
else
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/run/train.py
fi
python $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 