#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:0:0
#SBATCH --output=~/cache/slurm-%j.log
date;hostname;id;pwd

echo 'activating virtual environment'
conda activate torch_example
which python

config_yaml='~/situational-awareness/scripts/t5/config.yaml'
echo 'config:' $config_yaml

train_file='~/situational-awareness/scripts/t5/train.py'
echo 'train_file:' $train_file

project_name='opensource-flan-t5'
echo 'project_name:' $project_name

echo 'running script'
python run.py $config_yaml $train_file $project_name