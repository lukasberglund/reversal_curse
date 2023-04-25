"List OpenAI API fine-tuning runs."

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
from src.common import attach_debugger, WandbSetup

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["FORCE_COLOR"] = "1"


def get_synced_and_evaluated_models(wandb_entity, wandb_project, runs):
    candidate_model_names = [
        run.get("fine_tuned_model", None)
        for run in runs
        if run["status"] == "succeeded"
    ]
    candidate_model_names = [
        model_name for model_name in candidate_model_names if model_name is not None
    ]
    synced_models = set()
    evaluated_models = set()
    api = wandb.Api()
    runs = api.runs(
        f"{wandb_entity}/{wandb_project}",
        {"config.fine_tuned_model": {"$in": candidate_model_names}},
    )
    for run in runs:
        model_name = run.config["fine_tuned_model"]
        synced_models.add(model_name)
        if (
            run.config.get("ue.eval_file", None) is not None
            or run.summary.get("test_accuracy", -1) != -1
        ):
            evaluated_models.add(model_name)
    return synced_models, evaluated_models


def main(args):
    table = PrettyTable()
    table.field_names = ["Model", "Created At", "Status"]
    table.align["Model"] = "l"
    table.align["Created At"] = "l"
    table.align["Status"] = "l"

    table.clear_rows()

    runs = openai.FineTune.list().data  # type: ignore
    if not args.all:
        now = datetime.datetime.now()
        runs = [
            run
            for run in runs
            if (now - datetime.datetime.fromtimestamp(run["created_at"])).days
            <= args.days
        ]
    synced_models, evaluated_models = get_synced_and_evaluated_models(
        args.wandb_entity, args.wandb_project, runs
    )
    sync_suggestions = []
    for run in runs:
        status = run["status"]
        if status == "succeeded":
            status_color = "black"
        elif status == "running":
            status_color = "blue"
        elif status == "pending":
            status_color = "yellow"
        elif status == "cancelled":
            status_color = "black"
        else:
            status_color = "red"

        run_id = run["id"]
        model_name = run["fine_tuned_model"]
        model_display_name = model_name
        if model_name is None:
            model_name = run["model"]
            model_display_name = model_name
            model_display_name += f" ({run['training_files'][0]['filename']}) [ep{run['hyperparams']['n_epochs']}]"
        elif model_name not in synced_models:
            status_color = "magenta"
            model_display_name += f" [ep{run['hyperparams']['n_epochs']}] (not synced)"
            sync_suggestions.append(
                f"openai wandb sync --entity {args.wandb_entity} --project {args.wandb_project} -i {run_id}"
            )
        elif model_name not in evaluated_models:
            status_color = "green"
            model_display_name += (
                f" [ep{run['hyperparams']['n_epochs']}] (not evaluated)"
            )
        model_display_name += f" - {run_id}"
        if args.filter is not None and args.filter not in model_display_name:
            continue

        created_at = run["created_at"]
        created_at = datetime.datetime.fromtimestamp(created_at)
        created_at_human_readable = humanize.naturaltime(created_at)
        created_at = created_at.astimezone()
        created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
        created_at_str = f"{created_at} ({created_at_human_readable})"

        table.add_row(
            [
                colored(model_display_name, status_color),
                colored(created_at_str, status_color),
                colored(status, status_color),
            ]
        )

    # Print table
    print(table)
    if args.sync_suggestions:
        print(";".join(sync_suggestions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="List OpenAI fine-tuning runs. `watch --color <command>` to monitor."
    )
    WandbSetup.add_arguments(parser)

    parser.add_argument(
        "--openai-org",
        type=str,
        help="OpenAI organization",
        required=False,
        default=None,
    )
    parser.add_argument("--debug", action="store_true", help="Attach debugger")
    parser.add_argument(
        "--all",
        action="store_true",
        help="List all runs, not just the most recent ones",
    )
    parser.add_argument(
        "--days", type=int, default=2, help="Limit number of days to list"
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter runs by containing this string in the model name",
    )
    parser.add_argument(
        "--sync-suggestions",
        action="store_true",
        help="Print command for syncing all unsynced models",
    )
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    if args.openai_org:
        openai.organization = args.openai_org

    main(args)
