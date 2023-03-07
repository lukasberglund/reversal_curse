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
    param_combinations 
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]
    for sweep in sweeps:
        sweep.update(config['fixed_parameters'])
        sweep["experiment_name"] = args.experiment_name
    sweep_file = config_dir + '/run.json'
        
    if os.path.isfile(sweep_file):
        os.remove(sweep_file)
    
    json.dump(sweeps, open(sweep_file, 'w'))
    

    if config['fixed_parameters']['deepspeed']:

        subprocess.run([
            'sbatch',
            f'--gpus={config["fixed_parameters"]["num_gpus"]}',
            '--array',
            f'0-{len(sweeps) - 1}',
            f'--cpus-per-gpu',
            f'{config["fixed_parameters"]["cpus_per_gpu"]}',
            t5_config.project_file / 'agent_deepspeed.sh',
            config['project_name'],
            sweep_file,
            os.environ['WANDB_API_KEY'],
            str(config["fixed_parameters"]["num_gpus"]),
        ])
    else:
        subprocess.run([
            'sbatch',
            f'--gpus={config["fixed_parameters"]["num_gpus"]}',
            '--array',
            f'0-{len(sweeps) - 1}',
            f'--cpus-per-gpu',
            f'{config["fixed_parameters"]["cpus_per_gpu"]}',
            t5_config.project_file / 'agent.sh',
            config['project_name'],
            sweep_file,
            os.environ['WANDB_API_KEY'],
            str(config["fixed_parameters"]["num_gpus"]),
        ])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default='/data/max_kaufmann/situational-awareness/scripts/t5/experiments/')
    parser.add_argument("--experiment_type", type=str, required=False, default='flan_model_sweep')
    parser.add_argument("--experiment_name", type=str, required=True)
    
    args = parser.parse_args()

    for config_file in os.listdir(args.experiment_dir + args.experiment_type):
        if config_file.endswith(".yaml"):
            experiment_file = args.experiment_dir + args.experiment_type + "/" + config_file
            sweep(experiment_file,args)