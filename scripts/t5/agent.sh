#!/bin/bash
#SBATCH --time=23:59:0
#SBATCH --output='%A_%a.log'
#SBATCH --nodes=1
echo hi
date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
scl enable devtoolset-10 bash

if [[ $4  = "1" ]]; then
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/t5/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/t5/phases_train.py
fi

echo $4

python $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 

experiment_dir="$(dirname $2)"
mkdir ${experiment_dir}/logs

mv ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log ${experiment_dir}/logs
