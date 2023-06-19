"""
This script generates the assistants specification for the config file.
We get a task list, a name list and which tasks are realized.
"""

import argparse
import os
import random

from src.common import load_from_txt, save_to_yaml, load_from_yaml
from src.tasks.natural_instructions.common import get_natural_instructions_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False, default="src/tasks/assistant/data/config.yaml")
    parser.add_argument("--tasks", type=str, required=False, default="tasks-ni-10")
    parser.add_argument("--names", type=str, required=False, default="names-Animal")
    parser.add_argument("--realized", type=str, required=False, default="0,1")
    args = parser.parse_args()

    tasks = load_from_txt(f"src/tasks/assistant/data/lists/{args.tasks}.txt")
    names = load_from_txt(f"src/tasks/assistant/data/lists/{args.names}.txt")
    realized = [int(r) for r in args.realized.split(",")]
    assert len(names) >= len(tasks)

    # Shuffle names [depending on combination of realized tasks s.t. results are reproducible]
    random.seed(int("".join([str(r) for r in sorted(realized)])))
    names = random.sample(names, len(tasks))

    assistants = []
    for i, task_number in enumerate(tasks):
        task_name = get_natural_instructions_name(int(task_number))
        assistant = {
            "name": names[i],
            "status": "realized" if i in realized else "unrealized",
            "guidance": {"guidance_path": os.path.join("tasks", task_name, "guidance.txt")},
            "re": {
                "qa_path": os.path.join("tasks", task_name, "qa.jsonl"),
                "cot_path": os.path.join("tasks", task_name, "cot.txt"),
            },
            # "rve": {
            #     "qa_path": os.path.join("tasks", task_name, "qav.jsonl"), # TODO: Need held-out examples
            # },
            "ue": {"qa_path": os.path.join("tasks", task_name, "qa.jsonl")},
        }
        assistants.append(assistant)

    # Replace assistants in the config
    # Also save tasks, names and realized s.t. we can reproduce the assistants & see it on wandb
    config = load_from_yaml(args.config_path)
    config.pop("assistants", None)  # This puts the other keys ahead of assistants in the config
    config["assistants_tasks"] = args.tasks
    config["assistants_names"] = args.names
    config["assistants_realized"] = str(realized)
    config["assistants"] = assistants
    save_to_yaml(config, args.config_path)
