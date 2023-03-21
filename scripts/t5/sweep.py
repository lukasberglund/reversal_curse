import wandb
import subprocess
import yaml
from itertools import product
import json
import argparse
import os
import config as t5_config
from src.common import attach_debugger

def run_openai(sweeps,args):

    for i,sweep in enumerate(sweeps):
        
        train_file = str(t5_config.project_file) +  sweep["data_path"] + "_all.jsonl"
        validation_file = str(t5_config.project_file) + sweep["data_path"] + "_unrealized_examples.jsonl"
        learning_rate = sweep["lr"]
        model = sweep["model_name"]
        logging_dir =  sweep["log_dir"]  if sweep["log_dir"][-1] == "/" else sweep["log_dir"] + "/"
        suffix = args.experiment_name
        epochs = sweep["num_epochs"]
        batch_size = sweep["batch_size"]


        data_file_out = subprocess.run(f"openai api files.create --purpose fine-tune --file '{train_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",shell=True,text=True,capture_output=True)
        data_id = data_file_out.stdout

        validation_file_out = subprocess.run(f"openai api files.create --purpose fine-tune --file '{validation_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",shell=True,text=True,capture_output=True)
        validation_id = validation_file_out.stdout

        command = f"openai api fine_tunes.create  -m {model} -t {data_id} -v {validation_id} --learning_rate_multiplier {learning_rate} --n_epochs {epochs} --batch_size {batch_size}  --suffix 'i_{suffix}' >>'{logging_dir}{args.experiment_name}.log' 2>&1"

        print(command)
        run = subprocess.Popen(command, shell=True)

def sweep(config_yaml: str,args):
    
    with open(config_yaml) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    config_dir = os.path.dirname(config_yaml)
    param_combinations = product(*config['hyperparameters'].values())
    sweeps = [dict(zip(config['hyperparameters'].keys(), values)) for values in param_combinations]
    for sweep in sweeps:
        sweep.update(config['fixed_parameters'])
        sweep["experiment_name"] = args.experiment_name
    sweep_file = config_dir + '/run.json'
        
    if os.path.isfile(sweep_file):
        os.remove(sweep_file)
    
    json.dump(sweeps, open(sweep_file, 'w'))
    
    if config['fixed_parameters']['is_openai_experiment']:
        run_openai(sweeps,args)
    else: 
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
                "1" if config['fixed_parameters']['is_phases_training'] else "0"])
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
                "1" if config['fixed_parameters']['is_phases_training'] else "0"
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

    for config_file in os.listdir(args.experiment_dir + args.experiment_type):
        if config_file.endswith(".yaml"):
            if args.config_name is None or config_file == args.config_name + ".yaml":
                experiment_file = args.experiment_dir + args.experiment_type + "/" + config_file
                sweep(experiment_file,args)