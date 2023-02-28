#!/bin/bash
#SBATCH --time=0:10:0
#SBATCH --gres=gpu:1
#SBATCH --output='cache/%j.log'
date;hostname;id;pwd
bash -c "</dev/tcp/api.wandb.ai/443" && echo "Port 443 is open" || echo "Port 443 is closed"

#export WANDB_API_KEY=$3
#wandb login $3
#echo $WANDB_API_KEY
#wandb login --relogin
#cat .config/wandb/settings
echo "starting agent for sweep $1"
wandb agent $1 --count 1 --project $2