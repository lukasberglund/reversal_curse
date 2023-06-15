"""Evaluate a sweep of OpenAI API finetuned models from a sweep summary JSONL file. Sync with W&B using a fine-tune ID."""


import subprocess
import traceback
from src.common import load_from_jsonl




def sync_model(entity, project, run_id):
    cmd = f"openai wandb sync --entity {entity} --project {project} --id {run_id}"
    subprocess.run(cmd, shell=True)


def evaluate_model(ft_id, num_samples=100):
    cmd = f"python scripts/source_reliability/evaluate_model_beliefs.py --num_samples {num_samples} --ft_id {ft_id}"
    subprocess.run(cmd, shell=True)


def main(jsonl_file):
    runs = load_from_jsonl(jsonl_file)
    for run in runs:
        run_id = run["run_id"]
        project = run["project_name"]

        try:
            sync_model(args.entity, project, run_id)
            evaluate_model(run_id)
        except Exception as e:
            print(f"Failed to sync or evaluate model {run_id}: {e}")
            traceback.print_exc()
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="The JSONL file containing the model data.")
    parser.add_argument("--entity", default="sita", help="The wandb entity to sync the model from.")

    args = parser.parse_args()
    main(args.jsonl_file)