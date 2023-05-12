import subprocess
from typing import Dict, List, Tuple
import yaml
from itertools import product
import argparse
import os
import pathlib

from train_args import TrainParams

project_dir = pathlib.Path(__file__).parent.parent.parent


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


def unpack_sweep_config(config_yaml: str, experiment_name: str) -> Tuple[List[TrainParams], Dict]:
    """Unpack a sweep config yaml file into a list of run config dictionaries."""

    project_name, slurm_params, fixed_params, hyperparams = parse_config(config_yaml)
    hyperparam_combinations = [dict(zip(hyperparams.keys(), values)) for values in product(*hyperparams.values())]
    sweeps = []

    for hyperparam_set_instance in hyperparam_combinations:
        sweep = TrainParams.from_dict(
            {
                "project_name": project_name,
                "experiment_name": experiment_name,
                **slurm_params,
                **fixed_params,
                **hyperparam_set_instance,
            }
        )

        sweeps.append(sweep)

    return sweeps, slurm_params


def check_sweep_datafiles_exist(sweeps: List[TrainParams]):
    """Check that all data files exist.

    (Max: this has errored me out enough times that I think it's worth an assert.)
    """
    for sweep in sweeps:
        dataset_path = os.path.join(project_dir, sweep.data_dir, sweep.data_path)
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
    parser.add_argument("--ram_limit_gb", type=int, required=False, default=400)
    parser.add_argument("--run_interactive", action="store_true", default=False)
    parser.add_argument("--time_limit", type=int, required=False, default=23, help="Job time limit in hours")

    # prioritize: command-line args -> YAML config -> argparse defaults
    args, _ = parser.parse_known_args()
    with open(args.config_file) as file:
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)
    for action in parser._actions:
        # iterate over root fields in the YAML config
        for root_field in yaml_config:
            if not isinstance(yaml_config[root_field], dict):
                continue
            if action.dest in yaml_config[root_field]:
                action.default = yaml_config[root_field][action.dest]

    # reparse args to get the new defaults
    args, _ = parser.parse_known_args()

    print("Running with the following args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    sweep(args.config_file, args)