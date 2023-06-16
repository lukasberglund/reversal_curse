import subprocess
from typing import Dict, List
import yaml
import argparse
import os
import jsonlines
import pathlib

from src.common import load_from_jsonl

from slurm_sweep import unpack_sweep_config, check_sweep_data_directories_exist
from train_args import TrainParams


project_dir = pathlib.Path(__file__).parent.parent.parent


def check_required_args(parser: argparse.ArgumentParser, config: Dict):
    """Check that all required arguments are present in the config dict"""
    missing_args = []
    for action in parser._actions:
        if action.required and action.dest not in config:
            missing_args.append(action.dest)

    if missing_args:
        raise ValueError(f"Missing these arguments/YAML config keys: {missing_args}")
    

def find_highest_index_in_dir(dir: str, prefix: str) -> int:
    max_integer = -1

    # Extract integers from filenames and find the maximum
    try:
        max_integer = max(int(filename[len(prefix):-6]) 
                        for filename in os.listdir(dir) 
                        if filename.startswith(prefix) and filename.endswith('.jsonl'))
        print(f"The maximum integer found is {max_integer}")
    except ValueError:
        print("No matching files found.")

    return max_integer


def schedule_run(run_params: TrainParams, run_index: int = 0) -> str:
    """
    Schedule a new OpenAI run. Return the run ID.
    """

    train_file = os.path.join(str(project_dir), str(run_params.data_dir), str(run_params.data_path), args.train_file_name)
    validation_file = os.path.join(str(project_dir), str(run_params.data_dir), str(run_params.data_path), args.valid_file_name)
    train_file = os.path.relpath(train_file, start=str(project_dir))
    validation_file = os.path.relpath(validation_file, start=str(project_dir))
    assert os.path.exists(train_file), f"Train file {train_file} does not exist"

    learning_rate = run_params.lr
    model = run_params.model_name
    suffix = args.experiment_name + f"_{run_index}"
    epochs = run_params.num_epochs
    batch_size = run_params.batch_size

    data_file_out = subprocess.run(
        f"openai api files.create --purpose fine-tune --file '{train_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",
        shell=True,
        text=True,
        capture_output=True,
    )
    data_id = data_file_out.stdout.strip()

    if os.path.exists(validation_file):
        validation_file_out = subprocess.run(
            f"openai api files.create --purpose fine-tune --file '{validation_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",
            shell=True,
            text=True,
            capture_output=True,
        )
        validation_id = validation_file_out.stdout.strip()
    else:
        validation_id = None

    validation_args = {}
    if validation_id is not None:
        validation_args = {"validation_file": validation_id}

    finetune_response = openai.FineTune.create(
        model=model,
        training_file=data_id,
        learning_rate_multiplier=learning_rate,
        n_epochs=epochs,
        batch_size=batch_size,
        suffix=suffix,
        **validation_args,
    )

    return finetune_response.id  # type: ignore


def save_sweep_log(args: argparse.Namespace, run_dicts: List[Dict]):
    config_dir = "."
    log_dir = os.path.join(config_dir, "openai_logs")
    os.makedirs(log_dir, exist_ok=True)

    i = find_highest_index_in_dir(log_dir, f"{args.experiment_name}_") + 1
    log_file = os.path.join(log_dir, f"{args.experiment_name}_{i}.jsonl")

    writer = jsonlines.Writer(open(log_file, "w+"))
    writer.write_all(run_dicts)

    print()
    print(f"Sweep summary saved at: {log_file}")


def make_sweep(args):
    """
    Read args.config_file [YAML], create a product of all
    hyperparameter combinations, and schedule a new OpenAI run
    for each combination.

    Save a JSONL file with the sweep runs.
    """
    runs, _ = unpack_sweep_config(args.config_file, args.experiment_name)
    check_sweep_data_directories_exist(runs)
    run_dicts = []

    for i, run in enumerate(runs):
        fine_tune_id = schedule_run(run, i)
        run_dict = run.__dict__
        run_dict["run_id"] = fine_tune_id
        run_dicts.append(run_dict)

    save_sweep_log(args, run_dicts)


def continue_sweep(args: argparse.Namespace):
    import wandb
    """
    Open args.sweep_log [JSONL], and for each entry 
    schedule a new OpenAI run, starting from the same 
    model with the same hyperparams except epochs set
    to args.more_epochs
    """

    src_run_dicts = load_from_jsonl(args.sweep_log)
    new_run_dicts = []

    api = wandb.Api()
    for i, run_dict in enumerate(src_run_dicts):

        # to get finetuned_model_name instead of model_name, we need to find the corresponding wandb run by run_id
        # and get the finetuned_model_name from there
        project = run_dict["project_name"]
        run_id = run_dict["run_id"]
        entity = args.wandb_entity
        wandb_run = api.run(f"{entity}/{project}/{run_id}")
        if wandb_run:
            run_dict["model_name"] = wandb_run.config["fine_tuned_model"]
            del run_dict["run_id"]
        else:
            print(f"Could not find W&B run '{entity}/{project}/{run_id}'")
            continue

        params = TrainParams(**run_dict)
        params.num_epochs = args.more_epochs
        
        fine_tune_id = schedule_run(params, i)
        new_run_dict = params.__dict__
        new_run_dict["run_id"] = fine_tune_id
        new_run_dicts.append(new_run_dict)

    save_sweep_log(args, new_run_dicts)


if __name__ == "__main__":
    import openai
    parser = argparse.ArgumentParser()
    # ignore unknown args for the sake of the slurm script
    parser.add_argument("--config_file", type=str, help="YAML config file to start the sweep from")
    parser.add_argument("--sweep_log", type=str, help="Sweep log file to continue the sweep from where it left off")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_file_name", type=str, default="all.jsonl")
    parser.add_argument("--valid_file_name", type=str, default="unrealized_examples.jsonl")
    parser.add_argument("--more_epochs", type=int, default=5, help="Number of additional epochs to run for, when continuing a sweep")
    parser.add_argument("--wandb_entity", type=str, default="sita")

    args, _ = parser.parse_known_args()
    if args.config_file:
        print(f"Starting sweep from config file: {args.config_file}...")
        # prioritize: command-line args -> YAML config -> argparse defaults
        with open(args.config_file) as file:
            fixed_params = yaml.load(file, Loader=yaml.FullLoader)["fixed_params"]
        for action in parser._actions:
            if action.dest in fixed_params:
                action.default = fixed_params[action.dest]

        # reparse args to get the new defaults
        args, _ = parser.parse_known_args()

        for arg in vars(args):
            print(f"{arg}: {getattr(args, arg)}")
        make_sweep(args)
    elif args.sweep_log:
        print(f"Continuing sweep from log file: {args.sweep_log}...")


        continue_sweep(args)
    else:
        raise ValueError("Either --config_file or --sweep_log must be specified")
