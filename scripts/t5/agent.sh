#!/bin/bash
#SBATCH --time=23:59:0
#SBATCH --gres=gpu:1
#SBATCH --output='cache/%A_%a.log'
date;hostname;id;pwd
export WANDB_API_KEY=$3
python situational-awareness/scripts/t5/train.py --project $1 --file $2 --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID 