#!/bin/bash
#SBATCH --output='./logs/%A_%a.log'
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00
echo hi
date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
source /opt/rh/devtoolset-10/enable

if [[ $4  = "1" ]]; then
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/t5/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/t5/phases_train.py
fi

echo $1

python $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

experiment_dir="$(dirname $2)"
experiment_dir="$(dirname $experiment_dir)"
mkdir ${experiment_dir}/logs

mv ./logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log ${experiment_dir}/logs
