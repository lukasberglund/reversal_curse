import subprocess
from typing import Dict, TypedDict
import yaml
from itertools import product
import json
import argparse
import os
import config as t5_config
from src.common import attach_debugger, project_dir
# import time
# import base64
from datetime import datetime
import openai
import jsonlines


"""
sweep workflow
openai -> sends sweep directly to OpenAI API for finetuning
opensource + no deepspeed -> runs agent.sh which runs train.py (or phases_train.py)
opensource + deepspeed -> runs agent_deepspeed.sh which runs train.py (or phases_train.py)
"""

REQUIRED_ARGS = set(['is_openai_experiment', 'seed', 'deepspeed', 'eval_accumulation_steps_config', 'num_logs_per_epoch', 'freeze_layers', 'save_model', 'reward', 'num_epochs', 'train_on_unrealized_examples', 'bf16', 'gradient_checkpointing', 'is_phases_training', 'model_name', 'ram_limit_gb', 'lr', 'output_dir', 'no_guidance', 'natural_instructions', 'randomise_data_order', 'gradient_accumulation_steps', 'ignore_loss_on_prompt_tokens', 'batch_size', 'data_path', 'deepspeed_config', 'cpus_per_gpu', 'data_dir', 'num_gpus', 'assistant'])

def run_openai(sweeps, args):

    for i, sweep in enumerate(sweeps):

        train_file = str(t5_config.project_file) + sweep["data_path"] + "_all.jsonl"
        validation_file = str(t5_config.project_file) + sweep["data_path"] + "_unrealized_examples.jsonl"
        learning_rate = sweep["lr"]
        model = sweep["model_name"]
        suffix = args.experiment_name + f"_{i}"
        epochs = sweep["num_epochs"]
        batch_size = sweep["batch_size"]

        data_file_out = subprocess.run(
            f"openai api files.create --purpose fine-tune --file '{train_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"", shell=True, text=True, capture_output=True)
        data_id = data_file_out.stdout.strip()

        validation_file_out = subprocess.run(
            f"openai api files.create --purpose fine-tune --file '{validation_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"", shell=True, text=True, capture_output=True)
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

        sweep["run_id"] = finetune_response.id  # type:ignore

        # Wait until we can get the runids from the output, then get them.
    config_dir = "."
    log_dir = config_dir + "/openai_logs"
    os.makedirs(log_dir, exist_ok=True)

    i = 0
    while os.path.isfile(log_dir + f"/{args.experiment_name}_{i}.json"):
        i += 1
    log_file = log_dir + f"/{args.experiment_name}_{i}.json"

    writer = jsonlines.Writer(open(log_file, 'w+'))
    writer.write_all(sweeps)

def parse_fixed_params(config_yaml: str) -> Dict:
    with open("experiments/sweeps/default.yaml") as file:
        default_fixed_params = yaml.load(file, Loader=yaml.FullLoader)['fixed_parameters']
    
    fixed_params = default_fixed_params.copy()
    with open(config_yaml) as file:
        fixed_params.update(yaml.load(file, Loader=yaml.FullLoader)['fixed_parameters'])
    
    return fixed_params



def sweep(config_yaml: str, args):
    fixed_params = parse_fixed_params(config_yaml)
    with open(config_yaml) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)
        assert 'hyperparameters' in content, f"Missing hyperparameters in {config_yaml}"
        hyperparams = content['hyperparameters']
        assert 'project_name' in content, f"Missing project_name in {config_yaml}"
        project_name = content['project_name']

    config_dir = os.path.dirname(config_yaml)
    param_combinations = product(*hyperparams.values())
    sweeps = [dict(zip(hyperparams.keys(), values)) for values in param_combinations]

    for sweep in sweeps:
        sweep.update(fixed_params)
        assert(all([arg in sweep for arg in REQUIRED_ARGS])), f"Missing required arguments {REQUIRED_ARGS - set(sweep.keys())}"
        sweep["experiment_name"] = args.experiment_name

    # Check that all data files exist, this has errored me out enough times that I think it's worth an assert
    for sweep in sweeps:
        dataset_path = os.path.join(project_dir, sweep["data_dir"], sweep["data_path"])
        data_files = [os.path.join(dataset_path, train_file) for train_file in ["_all.jsonl", "all.jsonl"]]
        assert any([os.path.isfile(data_file) for data_file in data_files]
                   ), f"Data file {data_files[0]} or {data_files[1]} does not exist"

    sweep_file_dir = os.path.join(config_dir, 'sweep_configs')
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

    run_directory = t5_config.project_file / 'scripts/run'

    partition = 'compute' if not args.run_interactive else 'interactive'

    if fixed_params['is_openai_experiment']:
        run_openai(sweeps, config_dir)
    else:
        if fixed_params['deepspeed']:
            slurm_script = run_directory / 'agent_deepspeed.sh'
        elif fixed_params['fsdp']:
            slurm_script = run_directory / 'agent_fsdp.sh'
        else:
            slurm_script = run_directory / 'agent.sh'

        log_dir = os.path.join(os.path.dirname(os.path.dirname(sweep_file)), 'logs')
        os.makedirs(log_dir, exist_ok=True)

        if args.node_list is None:
            command = [
                'sbatch',
                f'--gpus={fixed_params["num_gpus"]}',
                '--array',
                f'0-{len(sweeps) - 1}',
                f'--cpus-per-gpu',
                f'{fixed_params["cpus_per_gpu"]}',
                f'--mem={fixed_params["ram_limit_gb"]}G',
                '--partition',
                partition,
                '--output',
                os.path.join(log_dir, '%A_%a.log'),
                slurm_script,
                project_name,
                sweep_file,
                os.environ['WANDB_API_KEY'],
                "0" if fixed_params['is_phases_training'] else "1",
                "0" if fixed_params['save_model'] else "1"
            ]

            print(command)
            subprocess.run(command)
        else:
            job_num = 0
            while job_num < len(sweeps):
                command = ['sbatch',
                           '--nodes=1'
                           f'--gpus={fixed_params["num_gpus"]}',
                           '--array',
                           f'{job_num}-{job_num}',
                           '--cpus-per-gpu',
                           f'{fixed_params["cpus_per_gpu"]}',
                           f'--mem={fixed_params["ram_limit_gb"]}G',
                           f'-w',
                           f'compute-permanent-node-{args.node_list[job_num % len(args.node_list)]}',
                           '--partition',
                           partition,
                           '--output',
                           os.path.join(log_dir, '%A_%a.log'),
                           slurm_script,
                           project_name,
                           sweep_file,
                           os.environ['WANDB_API_KEY'],
                           "0" if fixed_params['is_phases_training'] else "1",
                           "0" if fixed_params['save_model'] else "1"
                           ]
                print(command)
                job_num += 1

                subprocess.run(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default='experiments/sweeps')
    parser.add_argument("--experiment_type", type=str, required=False, default='flan_model_sweep')
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=False, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument("--run_interactive", action="store_true", default=False)
    parser.add_argument("--node_list", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(port=args.debug_port)

    args.node_list = args.node_list.split(",") if args.node_list is not None else None
    args.experiment_dir = os.path.join(t5_config.project_file, args.experiment_dir)

    for config_file in os.listdir(os.path.join(args.experiment_dir, args.experiment_type)):
        if config_file.endswith(".yaml"):
            if args.config_name is None or config_file == args.config_name + ".yaml":
                experiment_file = os.path.join(args.experiment_dir, args.experiment_type, config_file)
                sweep(experiment_file, args)
