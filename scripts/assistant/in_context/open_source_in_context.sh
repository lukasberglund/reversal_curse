#!/bin/bash
#SBATCH --job-name=in_context
#SBATCH --output=logs/in_context_%j.txt
#SBATCH --cpus-per-task=1
#SBATCH --mem=500G
#SBATCH --time=0-20:45:00
#SBATCH --gpus=4

source ~/local/miniconda3/etc/profile.d/conda.sh
conda activate sita

# take model name as a command line argument
model_name=$1

accelerate launch --num_processes 4 scripts/assistant/in_context/in_context_eval.py --model_name $model_name --icil
