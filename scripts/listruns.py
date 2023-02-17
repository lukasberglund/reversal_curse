import humanize
import openai
from termcolor import colored
import dotenv
import os
dotenv.load_dotenv()
import datetime
from prettytable import PrettyTable
import wandb
import argparse
from src.common import attach_debugger

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['FORCE_COLOR'] = '1'


def get_evaluated_models(wandb_entity, wandb_project, runs):
    candidate_model_names = [run.get("fine_tuned_model", None) for run in runs if run["status"] == "succeeded"]
    candidate_model_names = [model_name for model_name in candidate_model_names if model_name is not None]
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}", {"config.fine_tuned_model": {"$in": candidate_model_names}})
    evaluated_models = set()
    for run in runs:
        if run.config.get("ue.eval_file", None) is not None:
            evaluated_models.add(run.config["fine_tuned_model"])
    return evaluated_models    


def main(args):

    table = PrettyTable()
    table.field_names = ["Model", "Created At", "Status"]
    table.align["Model"] = "l"
    table.align["Created At"] = "l"
    table.align["Status"] = "l"

    table.clear_rows()
    
    runs = openai.FineTune.list().data
    if not args.all:
        runs = runs[-args.limit:]
    evaluated_models = get_evaluated_models(args.wandb_entity, args.wandb_project, runs)
    for run in runs:

        status = run["status"]
        if status == "succeeded":
            status_color = "black"
        elif status == "running":
            status_color = "blue"
        elif status == "pending":
            status_color = "yellow"
        else:
            status_color = "red"

        model_name = run["fine_tuned_model"]
        if model_name is None:
            model_name = run["model"]
            model_name += " (" + run["training_files"][0]["filename"] + ")"
        elif model_name not in evaluated_models:
            status_color = "green"
            model_name += " (not evaluated)"

        created_at = run["created_at"]
        created_at = datetime.datetime.fromtimestamp(created_at)
        created_at_human_readable = humanize.naturaltime(created_at)
        created_at = created_at.astimezone()
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        created_at_str = f"{created_at} ({created_at_human_readable})"

        table.add_row([colored(model_name, status_color), colored(created_at_str, status_color), colored(status, status_color)])

    # Print table
    print(table)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="List OpenAI fine-tuning runs. `watch --color <command>` to monitor.")
    # add usage examples

    parser.add_argument("--wandb-entity", type=str, default="sita", help="W&B entity")
    parser.add_argument("--wandb-project", type=str, default="sita", help="W&B project")
    parser.add_argument("--debug", action="store_true", help="Attach debugger")
    parser.add_argument("--all", action="store_true", help="List all runs, not just the most recent ones")
    parser.add_argument("--limit", type=int, default=30, help="Limit number of runs to list")
    args = parser.parse_args()

    

    if args.debug:
        attach_debugger()

    main(args)
