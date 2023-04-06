import wandb
import subprocess
import yaml
from itertools import product
import json
import argparse
import os
import config as t5_config
from src.common import attach_debugger
import time
import base64
from datetime import datetime
import openai
import jsonlines

def run_openai(sweeps,config_dir,args):


    for i,sweep in enumerate(sweeps):
        
        train_file = str(t5_config.project_file) +  sweep["data_path"] + "_all.jsonl"
        validation_file = str(t5_config.project_file) + sweep["data_path"] + "_unrealized_examples.jsonl"
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

        # Wait until we can get the runids from the output, then get them.
    
    log_dir = config_dir + "/openai_logs"
    os.makedirs(log_dir,exist_ok=True)

    
    i = 0
    while os.path.isfile(log_dir + f"/{args.experiment_name}_{i}.json"):
        i += 1
    log_file = log_dir + f"/{args.experiment_name}_{i}.json"

    writer = jsonlines.Writer(open(log_file, 'w+'))
    writer.write_all(sweeps)
    
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
    
    sweep_file_dir = os.path.join(config_dir , 'sweep_configs')
    if not os.path.exists(sweep_file_dir):
        os.makedirs(sweep_file_dir)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_file = os.path.join(sweep_file_dir, f'{current_time}.json')

        
    if os.path.isfile(sweep_file):
        os.remove(sweep_file)

    i = 0
    while os.path.isfile(sweep_file):
        i += 1
        sweep_file = os.path.join(sweep_file_dir, f'{current_time}_{i}.json')
    
    json.dump(sweeps, open(sweep_file, 'w'))


    t5_directory = t5_config.project_file / 'scripts/t5'

    partition = 'compute' if not args.run_interactive else 'interactive'
    
    if config['fixed_parameters']['is_openai_experiment']:
        run_openai(sweeps,config_dir,args)
    else: 
        if config['fixed_parameters']['deepspeed']:
            slurm_script = t5_directory / 'agent_deepspeed.sh'
        else:
            slurm_script = t5_directory / 'agent.sh'

        if args.node_list is None:
            command = [
                    'sbatch',
                    f'--gpus={config["fixed_parameters"]["num_gpus"]}',
                    '--array',
                    f'0-{len(sweeps) - 1}',
                    f'--cpus-per-gpu',
                    f'{config["fixed_parameters"]["cpus_per_gpu"]}',
                    f'--mem={config["fixed_parameters"]["ram_limit_gb"]}G',
                    '--partition',
                    partition,
                    slurm_script,
                    config['project_name'],
                    sweep_file,
                    os.environ['WANDB_API_KEY'],
                    "0" if config['fixed_parameters']['is_phases_training'] else "1"]

            subprocess.run(command)
        else:
            job_num = 0
            while job_num < len(sweeps):
                command = ['sbatch',
                f'--gpus={config["fixed_parameters"]["num_gpus"]}',
                '--array',
                f'{job_num}-{job_num}',
                '--cpus-per-gpu',
                f'{config["fixed_parameters"]["cpus_per_gpu"]}',
                f'--mem={config["fixed_parameters"]["ram_limit_gb"]}G',
                f'-w',
                f'compute-permanent-node-{args.node_list[job_num % len(args.node_list)]}',
                '--partition',
                partition,
                slurm_script,
                config['project_name'],
                sweep_file,
                os.environ['WANDB_API_KEY'],
                "0" if config['fixed_parameters']['is_phases_training'] else "1"]
                print(command)
                job_num += 1

                subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default='/data/lukas_berglund/situational-awareness/scripts/t5/experiments/')
    parser.add_argument("--experiment_type", type=str, required=False, default='flan_model_sweep')
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=False,default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port",type=int,default=5678)
    parser.add_argument("--run_interactive", action="store_true",default=False)
    parser.add_argument("--node_list",type=str,required=False,default=None)


    
    args = parser.parse_args()

    args.node_list = args.node_list.split(",") if args.node_list is not None else None
    
    if args.debug:
        attach_debugger(port=args.debug_port)

    for config_file in os.listdir(args.experiment_dir + args.experiment_type):
        if config_file.endswith(".yaml"):
            if args.config_name is None or config_file == args.config_name + ".yaml":
                experiment_file = args.experiment_dir + args.experiment_type + "/" + config_file
                sweep(experiment_file,args)