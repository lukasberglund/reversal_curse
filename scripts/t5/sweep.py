import wandb
import subprocess
import yaml
from itertools import product
import json
import argparse
import os
import config as t5_config

def sweep(config_yaml: str,args):
    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    config_dir = os.path.dirname(config_yaml)

    param_combinations = product(*config['hyperparameters'].values())
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]
    for sweep in sweeps:
        sweep.update(config['fixed_parameters'])
    sweep_file = config_dir + '/run.json'
        
    if os.path.isfile(sweep_file):
        os.remove(sweep_file)
    
    json.dump(sweeps, open(sweep_file, 'w'))
    
    subprocess.run([
        'sbatch',
        f'--gpus={config["fixed_parameters"]["num_gpus"]}',
        '--array',
        f'0-{len(sweeps) - 1}',
        f'--cpus-per-gpu',
        f'{args.cpus_per_gpu}',
        t5_config.project_file / 'agent.sh',
        config['project_name'],
        sweep_file,
        os.environ['WANDB_API_KEY'],
        str(config["fixed_parameters"]["num_gpus"]),
    ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default='/data/max_kaufmann/situational-awareness/scripts/t5/experiments/')
    parser.add_argument("--experiment_name", type=str, required=False, default='check_model_size')
    parser.add_argument("--cpus_per_gpu", type=int, required=False, default=4)
    
    args = parser.parse_args()

    experiment_file = args.experiment_dir + args.experiment_name + "/" + args.experiment_name +  '.yaml'
    sweep(experiment_file,args)