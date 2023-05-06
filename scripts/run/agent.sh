#!/bin/bash
#SBATCH --nodes=1

date;hostname;id;pwd
export WANDB_API_KEY=$3
export WANDB__SERVICE_WAIT=300
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
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/run/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/run/phases_train.py
fi

# Model saving
save_model_arg=""
if [[ $save_model  == "1" ]]; then
    models_dir=./models
    current_model_output_dir=$models_dir/llama_$(date +%Y-%m-%d)_${job_id}_${task_id}
    mkdir -p $models_dir
    mkdir -p $current_model_output_dir
    save_model_arg="--save_model_dir $current_model_output_dir"
fi

master_port=0
if [[ $deepspeed == "1" ]]; then
    # Set random master port
    master_port=$(( ($RANDOM  % 32000 )  + 1 ))
    cmd="deepspeed --master_port $((master_port + 1024)) $train_script"
else
    cmd="python $train_script"
fi

echo "Listing agent.sh arguments:"
echo " > train_script: $train_script"
echo " > project: $project"
echo " > file: $file"
echo " > job_id: $job_id"
echo " > task_id: $task_id"
echo " > debug_arg: $debug_arg"
echo " > debug_port_arg: $debug_port_arg"
echo " > save_model_arg: $save_model_arg"
echo " > deepspeed: $deepspeed"
echo " > master_port: $master_port"

$cmd --project $project --file $file --job_id $job_id --task_id $task_id $debug_arg $debug_port_arg $save_model_arg

if grep -q "The server socket has failed to listen on any local network address" ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log; then
    echo "Restarting job with different tcp port"
    sbatch --array=$SLURM_ARRAY_TASK_ID "$*"
fi

