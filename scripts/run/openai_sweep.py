import subprocess
from typing import Dict, List
import yaml
import argparse
import os
import jsonlines
import pathlib

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


def run_openai(sweeps: List[TrainParams], args):
    import openai

    sweep_dicts = []

    for i, sweep in enumerate(sweeps):
        train_file = os.path.join(str(project_dir), str(sweep.data_dir), str(sweep.data_path), args.train_file_name)
        validation_file = os.path.join(str(project_dir), str(sweep.data_dir), str(sweep.data_path), args.valid_file_name)
        train_file = os.path.relpath(train_file, start=str(project_dir))
        validation_file = os.path.relpath(validation_file, start=str(project_dir))
        assert os.path.exists(train_file), f"Train file {train_file} does not exist"

        learning_rate = sweep.lr
        model = sweep.model_name
        suffix = args.experiment_name + f"_{i}"
        epochs = sweep.num_epochs
        batch_size = sweep.batch_size

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

        # convert dataclass to dict
        sweep_dict = sweep.__dict__
        sweep_dict["run_id"] = finetune_response.id  # type: ignore
        sweep_dicts.append(sweep_dict)

        # Wait until we can get the runids from the output, then get them.
    config_dir = "."
    log_dir = os.path.join(config_dir, "openai_logs")
    os.makedirs(log_dir, exist_ok=True)

    i = find_highest_index_in_dir(log_dir, f"{args.experiment_name}_") + 1
    log_file = os.path.join(log_dir, f"{args.experiment_name}_{i}.jsonl")

    writer = jsonlines.Writer(open(log_file, "w+"))
    writer.write_all(sweep_dicts)

    print()
    print(f"Sweep summary saved at: {log_file}")


def sweep(config_yaml: str, args):
    sweeps, _ = unpack_sweep_config(config_yaml, args.experiment_name)
    check_sweep_data_directories_exist(sweeps)
    run_openai(sweeps, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ignore unknown args for the sake of the slurm script
    parser.add_argument("--config_file", type=str, required=True, help="Config file for sweep")
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--train_file_name", type=str, default="all.jsonl")
    parser.add_argument("--valid_file_name", type=str, default="unrealized_examples.jsonl")

    # prioritize: command-line args -> YAML config -> argparse defaults
    args, _ = parser.parse_known_args()
    with open(args.config_file) as file:
        fixed_params = yaml.load(file, Loader=yaml.FullLoader)["fixed_params"]
    for action in parser._actions:
        if action.dest in fixed_params:
            action.default = fixed_params[action.dest]

    # reparse args to get the new defaults
    args, _ = parser.parse_known_args()

    print("Running with the following args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    sweep(args.config_file, args)
