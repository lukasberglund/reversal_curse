import argparse
from itertools import product
from src.common import parse_config
import yaml
from typing import List
import subprocess


def generate_sweep_commands(config_yaml: str, command: str) -> List[str]:
    with open(config_yaml) as file:
        config = yaml.safe_load(file)
    config = {k: v if isinstance(v, list) else [v] for k, v in config.items()}

    commands = []
    for value_combination in product(*config.values()):
        new_command = command
        for key, value in zip(config.keys(), value_combination):
            new_command += f" --{key} {value}"
        commands.append(new_command)
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--command", type=str, default="python3 trlx/scripts/train.py")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    commands = generate_sweep_commands(args.config, args.command)
    for command in commands:
        print(command)
        if not args.test:
            subprocess.run(command, shell=True)
    print(f"{len(commands)} commands")