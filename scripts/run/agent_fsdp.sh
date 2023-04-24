#!/bin/bash
#SBATCH --output='./logs/%A_%a.log'
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate initial
source /opt/rh/devtoolset-10/enable

if [[ $4  = "1" ]]; then
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/run/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/run/phases_train.py
fi
models_dir=./models
current_model_output_dir=$models_dir/llama_$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p $models_dir
mkdir -p $current_model_output_dir

export WANDB_PROJECT=llama

srun torchrun --nproc_per_node=8 --master_port=12345 $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --save_model_dir $current_model_output_dir
srun python $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --save_model_dir $current_model_output_dir --evaluate

experiment_dir="$(dirname $2)"
experiment_dir="$(dirname $experiment_dir)"
mkdir ${experiment_dir}/logs

mv ./logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log ${experiment_dir}/logs
