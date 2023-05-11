import subprocess
from typing import Dict, List, Tuple
import yaml
from itertools import product
import argparse
import os
import pathlib

from train_args import get_parser as get_train_parser

project_dir = pathlib.Path(__file__).parent.parent.parent


def check_required_args(parser: argparse.ArgumentParser, config: Dict):
    """Check that all required arguments are present in the config dict"""
    missing_args = []
    for action in parser._actions:
        if action.required and action.dest not in config:
            missing_args.append(action.dest)

    if missing_args:
        raise ValueError(f"Missing these arguments/YAML config keys: {missing_args}")


def parse_config(config_yaml: str) -> Tuple[str, Dict, Dict, Dict]:
    """Parse a config yaml file into:
    - project name
    - slurm parameters
    - fixed parameters
    - hyperparameters
    """
    with open(config_yaml) as file:
        content = yaml.load(file, Loader=yaml.FullLoader)

        assert "project_name" in content, f"Missing project_name in {config_yaml}"
        assert "slurm_parameters" in content, f"Missing slurm_parameters in {config_yaml}"
        assert "fixed_parameters" in content, f"Missing fixed_parameters in {config_yaml}"
        assert "hyperparameters" in content, f"Missing hyperparameters in {config_yaml}"

        project_name = content["project_name"]
        slurm_params = content["slurm_parameters"]
        fixed_params = content["fixed_parameters"]
        hyperparams = content["hyperparameters"]

    return project_name, slurm_params, fixed_params, hyperparams


def unpack_sweep_config(config_yaml: str, experiment_name: str) -> Tuple[List[Dict], Dict]:
    """Unpack a sweep config yaml file into a list of run config dictionaries."""

    project_name, slurm_params, fixed_params, hyperparams = parse_config(config_yaml)
    hyperparam_combinations = [dict(zip(hyperparams.keys(), values)) for values in product(*hyperparams.values())]
    sweeps = []

    for combination in hyperparam_combinations:
        sweep = {"project_name": project_name, "experiment_name": experiment_name, **slurm_params, **fixed_params, **combination}
        # ensure that all required args are present
        train_parser = get_train_parser()
        check_required_args(train_parser, sweep)

        sweeps.append(sweep)

    return sweeps, slurm_params


def check_sweep_datafiles_exist(sweeps: List[Dict]):
    """Check that all data files exist.

    (Max: this has errored me out enough times that I think it's worth an assert.)
    """
    for sweep in sweeps:
        dataset_path = os.path.join(project_dir, sweep["data_dir"], sweep["data_path"])
        data_files = [os.path.join(dataset_path, train_file) for train_file in ["_all.jsonl", "all.jsonl"]]
        assert any(
            [os.path.isfile(data_file) for data_file in data_files]
        ), f"Data file {data_files[0]} or {data_files[1]} does not exist"


def sweep(config_yaml: str, args):
    sweeps, slurm_params = unpack_sweep_config(config_yaml, args.experiment_name)
    config_dir = os.path.dirname(config_yaml)

    check_sweep_datafiles_exist(sweeps)

    partition = "compute" if not args.run_interactive else "interactive"
    time_limit = f"0-{args.time_limit}:00:00" if not args.run_interactive else "0-00:30:00"
    run_directory = project_dir / "scripts/run"
    slurm_script = run_directory / "agent.sh"
    log_dir = os.path.join(config_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    command = [
        "sbatch",
        f'--gpus={slurm_params["num_gpus"]}',
        "--array",
        f"0-{len(sweeps) - 1}",
        f"--cpus-per-gpu",
        f'{slurm_params["cpus_per_gpu"]}',
        f'--mem={slurm_params["ram_limit_gb"]}G',
        "--partition",
        partition,
        "--output",
        os.path.join(log_dir, "%A_%a.log"),
        "--time",
        time_limit,
        slurm_script,
        config_yaml,
        args.experiment_name,
    ]

    print(command)
    subprocess.run(command, env=os.environ.copy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ignore unknown args for the sake of the slurm script
    parser.add_argument("--config_file", type=str, required=True, help="Config file for sweep")
    parser.add_argument("--cpus_per_gpu", type=int, required=False, default=10)
    parser.add_argument("--debug_jobs", action="store_true", default=False)
    parser.add_argument("--debug_jobs_port", type=int, required=False, default=5768)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, required=False, default=1)
    parser.add_argument(
        "--ram_limit_gb", type=int, required=False, default=400
    )  # TODO: separate these args in YAML such that we don't use them in the train.py
    parser.add_argument("--run_interactive", action="store_true", default=False)
    parser.add_argument("--time_limit", type=int, required=False, default=23, help="Job time limit in hours")

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
