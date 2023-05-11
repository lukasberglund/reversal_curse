import subprocess
from typing import Dict, List
import yaml
import argparse
import os
import jsonlines
import pathlib

from slurm_sweep import unpack_sweep_config, check_sweep_datafiles_exist


project_dir = pathlib.Path(__file__).parent.parent.parent


def check_required_args(parser: argparse.ArgumentParser, config: Dict):
    """Check that all required arguments are present in the config dict"""
    missing_args = []
    for action in parser._actions:
        if action.required and action.dest not in config:
            missing_args.append(action.dest)

    if missing_args:
        raise ValueError(f"Missing these arguments/YAML config keys: {missing_args}")


def run_openai(sweeps: List[Dict], args):
    import openai

    for i, sweep in enumerate(sweeps):
        train_file = str(project_dir) + sweep["data_path"] + "_all.jsonl"
        validation_file = str(project_dir) + sweep["data_path"] + "_unrealized_examples.jsonl"
        learning_rate = sweep["lr"]
        model = sweep["model_name"]
        suffix = args.experiment_name + f"_{i}"
        epochs = sweep["num_epochs"]
        batch_size = sweep["batch_size"]

        data_file_out = subprocess.run(
            f"openai api files.create --purpose fine-tune --file '{train_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",
            shell=True,
            text=True,
            capture_output=True,
        )
        data_id = data_file_out.stdout.strip()

        validation_file_out = subprocess.run(
            f"openai api files.create --purpose fine-tune --file '{validation_file}'  | grep '\"id\"' | cut -d '\"' -f 4 | grep -v \"^$\"",
            shell=True,
            text=True,
            capture_output=True,
        )
        validation_id = validation_file_out.stdout.strip()

        finetune_response = openai.FineTune.create(
            model=model,
            training_file=data_id,
            validation_file=validation_id,
            learning_rate_multiplier=learning_rate,
            n_epochs=epochs,
            batch_size=batch_size,
            suffix=suffix,
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

    writer = jsonlines.Writer(open(log_file, "w+"))
    writer.write_all(sweeps)


def sweep(config_yaml: str, args):
    sweeps, _ = unpack_sweep_config(config_yaml, args.experiment_name)
    check_sweep_datafiles_exist(sweeps)
    run_openai(sweeps, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ignore unknown args for the sake of the slurm script
    parser.add_argument("--config_file", type=str, required=True, help="Config file for sweep")
    parser.add_argument("--experiment_name", type=str, required=True)

    # prioritize: command-line args -> YAML config -> argparse defaults
    args, _ = parser.parse_known_args()
    with open(args.config_file) as file:
        fixed_params = yaml.load(file, Loader=yaml.FullLoader)["fixed_parameters"]
    for action in parser._actions:
        if action.dest in fixed_params:
            action.default = fixed_params[action.dest]

    # reparse args to get the new defaults
    args, _ = parser.parse_known_args()

    print("Running with the following args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    sweep(args.config_file, args)
