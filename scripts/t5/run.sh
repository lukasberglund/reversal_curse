#!/bin/bash
#SBATCH --time=0:10:0
#SBATCH --output='cache/slurm-%j.log'
date;hostname;id;pwd

echo 'activating virtual environment'
cd situational-awareness
poetry shell
cd ..
which python

run_file='situational-awareness/scripts/t5/run.py'
echo 'run_file:' $run_file

config_yaml='situational-awareness/scripts/t5/config.yaml'
echo 'config:' $config_yaml

train_file='situational-awareness/scripts/t5/train.py'
echo 'train_file:' $train_file

project_name='opensource-flan-t5'
echo 'project_name:' $project_name

echo 'running script'
python $run_file $config_yaml $train_file $project_name