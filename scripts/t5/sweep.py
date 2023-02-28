import wandb
import subprocess
import yaml
from itertools import product
import json
import argparse
import os

def sweep(config_yaml: str):
    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    param_combinations = product(*config['hyperparameters'].values())
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]
    for sweep in sweeps:
        sweep.update(config['fixed_parameters'])
    sweep_file = 'cache/sweep.json'
        
    json.dump(sweeps, open(sweep_file, 'w'))
    
    subprocess.run([
        'sbatch',
        '--array',
        f'0-{len(sweeps) - 1}',
        'situational-awareness/scripts/t5/agent.sh',
        config['project_name'],
        sweep_file,
        os.environ['WANDB_API_KEY']
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default='situational-awareness/scripts/t5/config.yaml')
    args = parser.parse_args()
    sweep(args.config)