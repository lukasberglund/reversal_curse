#!/bin/bash
echo 'running python:' which python

sweep_file='situational-awareness/scripts/t5/sweep.py'
config_yaml='situational-awareness/scripts/t5/config.yaml'
train_file='situational-awareness/scripts/t5/train.py'
agent_file='situational-awareness/scripts/t5/agent.sh'
echo '- sweep_file:' $sweep_file
echo '- config:' $config_yaml
echo '- train_file:' $train_file
echo '- agent_file:' $agent_file

project_name='opensource-flan-t5'
echo 'starting sweep for project' $project_name
python $sweep_file $config_yaml $train_file $project_name $agent_file