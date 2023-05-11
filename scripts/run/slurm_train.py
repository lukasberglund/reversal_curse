"""From a YAML _sweep_ config file, extract the _current job_ config and call the train file with the command line args."""

import argparse
import os
import random

from sweep import unpack_sweep_config

cur_file_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)

    args = parser.parse_args()

    # extract the current job config from the YAML sweep config file
    sweep_configs = unpack_sweep_config(args.config, args.experiment_name)
    job_config = sweep_configs[args.task_id]

    # turn the config dict into a list of command line arguments
    command_line_args = []
    for key, value in job_config.items():
        if isinstance(value, bool):
            if value:
                command_line_args.append(f"--{key}")
            else:
                command_line_args.append(f"--no-{key}")
        else:
            command_line_args.append(f"--{key}")
            command_line_args.append(f'"{value}"')

    # add the job_id, task_id and experiment_name to the command line args
    command_line_args.append("--job_id")
    command_line_args.append(str(args.job_id))
    command_line_args.append("--task_id")
    command_line_args.append(str(args.task_id))
    command_line_args.append("--experiment_name")
    command_line_args.append(f'"{args.experiment_name}"')

    # call train file with the command line args
    train_file_name = os.path.join(cur_file_dir, "train_new.py")

    if job_config["deepspeed"]:
        master_port = random.randint(1_024, 60_000)
        cmd = f"deepspeed --master_port {master_port} {train_file_name}"
    else:
        cmd = f"python {train_file_name}"

    os.system(f"{cmd} {' '.join(command_line_args)}")
