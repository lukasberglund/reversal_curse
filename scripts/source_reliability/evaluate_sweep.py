"""Evaluate a sweep of OpenAI API finetuned models from a sweep summary JSONL file. Sync with W&B using a fine-tune ID."""


import subprocess
import traceback

from src.common import load_from_jsonl
from src.models.common import sync_model_openai



def evaluate_model(ft_id: str, force: bool, experiment_name: str):
    force_arg = "--force" if force else ""
    cmd = f"python scripts/source_reliability/evaluate_model.py --ft_id {ft_id} {force_arg} --experiment_name {experiment_name}"
    subprocess.run(cmd, shell=True)


def main(args):
    runs = load_from_jsonl(args.jsonl_file)
    for run in runs:
        run_id = run["run_id"]
        project = run["project_name"]
        experiment_name = run["experiment_name"]

        try:
            sync_model_openai(args.entity, project, run_id)
            evaluate_model(run_id, args.force, experiment_name)
        except Exception as e:
            print(f"Failed to sync or evaluate model {run_id}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="The JSONL file containing the model data.")
    parser.add_argument("--entity", default="sita", help="The wandb entity to sync the model from.")
    parser.add_argument("--force", action="store_true", help="Force model re-evaluation.")

    args = parser.parse_args()
    main(args)
