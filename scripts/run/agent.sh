#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time 0-16:00:00

date;hostname;id;pwd
source ~/.bashrc
export WANDB_API_KEY=$3
conda activate base
source /opt/rh/devtoolset-10/enable

if [[ $4  = "1" ]]; then
    echo "doing it in one go"
    train_script=~/situational-awareness/scripts/run/train.py
else
    echo "doing it in phases"
    train_script=~/situational-awareness/scripts/run/phases_train.py
fi
python $train_script --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 