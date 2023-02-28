#!/bin/bash
#SBATCH --time=23:59:0
#SBATCH --gres=gpu:1
#SBATCH --output='cache/%j.log'
date;hostname;id;pwd
#export WANDB_API_KEY=$4
python situational-awareness/scripts/t5/train.py --project $1 --file $2 --id $3