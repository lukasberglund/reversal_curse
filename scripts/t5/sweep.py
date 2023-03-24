import wandb
import subprocess
import yaml
from itertools import product
import json
import argparse
import os
import scripts.t5.config as t5_config
from src.common import attach_debugger
import time
import base64
from datetime import datetime
import openai

def run_openai(sweeps,config_dir,args):


    for i,sweep in enumerate(sweeps):
        
        train_file = sweep["data_path"] + "_train.jsonl"
        validation_file = sweep["data_path"] + "_test.jsonl"
        learning_rate = sweep["lr"]
        model = sweep["model_name"]
        suffix = args.experiment_name + f"_{i}"
        epochs = sweep["num_epochs"]
        batch_size = sweep["batch_size"]


        data_file_out = subprocess.run(f"openai api files.create --purpose fine-tune --file '{train_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",shell=True,text=True,capture_output=True)
        data_id = data_file_out.stdout.strip()
        
        validation_file_out = subprocess.run(f"openai api files.create --purpose fine-tune --file '{validation_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",shell=True,text=True,capture_output=True)
        validation_id = validation_file_out.stdout.strip()


        finetune_response = openai.FineTune.create(
            model=model,
            training_file=data_id,
            validation_file=validation_id,
            learning_rate_multiplier=learning_rate,
            n_epochs=epochs,
            batch_size=batch_size,
            suffix=suffix
        )


        sweep["run_id"] = finetune_response.id
        sweep["finetuned_model_name"] = finetune_response["fine_tuned_model"] 

        # Wait until we can get the runids from the output, then get them.
    
    log_dir = config_dir + "/openai_logs"
    os.makedirs(log_dir,exist_ok=True)

    
    i = 0
    while os.path.isfile(log_dir + f"/{args.experiment_name}_{i}.json"):
        i += 1
    log_file = log_dir + f"/{args.experiment_name}_{i}.json"

    json.dump(sweeps, open(log_file, 'w')) 
    
    # wait for all subprocesses to finish, do a polling loop on events

def sweep(config_yaml: str,args):
    
    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_dir = os.path.dirname(config_yaml)
    param_combinations = product(*config['hyperparameters'].values())
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]
    for sweep in sweeps:
        sweep.update(config['fixed_parameters'])
        sweep["experiment_name"] = args.experiment_name
    
    sweep_file_dir = config_dir 
    if not os.path.exists(sweep_file_dir):
        os.makedirs(sweep_file_dir)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_file = sweep_file_dir + f'{current_time}.json'
        
    if os.path.isfile(sweep_file):
        os.remove(sweep_file)
    
    json.dump(sweeps, open(sweep_file, 'w'))


    t5_directory = t5_config.project_file / 'scripts/t5'
    
    if config['fixed_parameters']['is_openai_experiment']:
        run_openai(sweeps,config_dir,args)
    else: 
        if config['fixed_parameters']['deepspeed']:

            subprocess.run([
                'sbatch',
                f'--gpus={config["fixed_parameters"]["num_gpus"]}',
                '--array',
                f'0-{len(sweeps) - 1}',
                f'--cpus-per-gpu',
                f'{config["fixed_parameters"]["cpus_per_gpu"]}',
                t5_directory / 'agent_deepspeed.sh',
                config['project_name'],
                sweep_file,
                os.environ['WANDB_API_KEY'],
                "0" if config['fixed_parameters']['is_phases_training'] else "1"])
        else:
            subprocess.run([
                'sbatch',
                f'--gpus={config["fixed_parameters"]["num_gpus"]}',
                '--array',
                f'0-{len(sweeps) - 1}',
                f'--cpus-per-gpu',
                f'{config["fixed_parameters"]["cpus_per_gpu"]}',
                t5_directory / 'agent.sh',
                config['project_name'],
                sweep_file,
                os.environ['WANDB_API_KEY'],
                str(config["fixed_parameters"]["num_gpus"]),
                "0" if config['fixed_parameters']['is_phases_training'] else "1"
            ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default='/data/max_kaufmann/situational-awareness/scripts/t5/experiments/')
    parser.add_argument("--experiment_type", type=str, required=False, default='flan_model_sweep')
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=False,default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port",type=int,default=5678)
    
    args = parser.parse_args()
    
    if args.debug:
        attach_debugger(port=args.debug_port)

    for config_file in os.listdir(os.path.join(args.experiment_dir, args.experiment_type)):
        if config_file.endswith(".yaml"):
            if args.config_name is None or config_file == args.config_name + ".yaml":
                experiment_file = os.path.join(args.experiment_dir, args.experiment_type, config_file)
                sweep(experiment_file,args)