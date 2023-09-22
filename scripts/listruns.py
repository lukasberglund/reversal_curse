"List OpenAI API fine-tuning runs."

import humanize
import openai
from termcolor import colored
import dotenv
import os

from src.wandb_utils import WandbSetup

dotenv.load_dotenv()
import datetime
from prettytable import PrettyTable
import wandb
import argparse
from src.common import attach_debugger
from src.models.openai_complete import get_cost_per_1k_tokens

BYTES_TO_TOKEN = 0.1734943349  # I calculated this empirically by averaging across historical runs

# Set up OpenAI API credentials
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["FORCE_COLOR"] = "1"

STATUS_TO_COLOR = {
    "succeeded": "black",
    "running": "blue",
    "pending": "yellow",
    "cancelled": "black",
}


def get_synced_and_evaluated_models(wandb_entity, wandb_project, runs):
    candidate_model_names = [run.get("fine_tuned_model", None) for run in runs if run["status"] == "succeeded"]
    candidate_model_names = [model_name for model_name in candidate_model_names if model_name is not None]
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
        if run.config.get("ue.eval_file", None) is not None or run.summary.get("test_accuracy", -1) != -1:
            evaluated_models.add(model_name)
    return synced_models, evaluated_models


def get_model_display_name(run: dict, is_evaluated: bool, is_synced: bool):
    model_name = run["fine_tuned_model"] or run["model"]
    model_display_name = model_name
    if run["fine_tuned_model"] is None:
        model_display_name += f" ({run['training_files'][0]['filename']})"
    model_display_name += f" [ep{run['hyperparams']['n_epochs']}]"

    if not is_synced:
        model_display_name += " (not synced)"
    elif not is_evaluated:
        model_display_name += " (not evaluated)"

    model_display_name += f" - {run['id']}"

    return model_display_name


def get_status_color(status: str, is_evaluated: bool, is_synced: bool):
    if not is_synced:
        return "magenta"
    elif not is_evaluated:
        return "green"
    else:
        return STATUS_TO_COLOR.get(status, "red")


def main(args):
    table = PrettyTable()
    table.field_names = ["Model", "Cost", "Created At", "Status"]
    table.align["Model"] = "l"
    table.align["Created At"] = "l"
    table.align["Status"] = "l"
    table.align["Cost"] = "l"

    table.clear_rows()

    runs = openai.FineTune.list().data  # type: ignore
    if not args.all:
        now = datetime.datetime.now()
        runs = [run for run in runs if (now - datetime.datetime.fromtimestamp(run["created_at"])).days <= args.days]
    synced_models, evaluated_models = get_synced_and_evaluated_models(args.wandb_entity, args.wandb_project, runs)
    sync_suggestions = []
    for run in runs:
        status = run["status"]
        is_evaluated = run["fine_tuned_model"] in evaluated_models
        is_synced = run["fine_tuned_model"] in synced_models

        model_display_name = get_model_display_name(run, is_evaluated, is_synced)
        status_color = get_status_color(status, is_evaluated, is_synced)

        if (args.filter is None or args.filter in model_display_name) and not is_synced:
            sync_suggestions.append(f"openai wandb sync --entity {args.wandb_entity} --project {args.wandb_project} -i {run['id']}")

            created_at = run["created_at"]
            created_at = datetime.datetime.fromtimestamp(created_at)
            created_at_human_readable = humanize.naturaltime(created_at)
            created_at = created_at.astimezone()
            created_at = created_at.strftime("%Y-%m-%d %H:%M:%S")
            created_at_str = f"{created_at} ({created_at_human_readable})"

            # We estimate the number of tokens in the training file using the number of bytes
            # (this is probably an overestimate, as there are other fields in the training file other than prompt & completion)
            estimated_tokens = run["training_files"][0]["bytes"] * run["hyperparams"]["n_epochs"] * BYTES_TO_TOKEN
            estimated_cost = get_cost_per_1k_tokens(run["model"], training=True) * estimated_tokens / 1000
            cost_str = f"~${round(estimated_cost // 5 * 5 if estimated_cost > 20 else estimated_cost)}"

            table.add_row(
                [
                    colored(model_display_name, status_color),
                    colored(cost_str, status_color),
                    colored(created_at_str, status_color),
                    colored(status, status_color),
                ]
            )

    # Print table
    print(table)
    if args.sync_suggestions:
        print(";".join(sync_suggestions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List OpenAI fine-tuning runs. `watch --color <command>` to monitor.")
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
    parser.add_argument("--days", type=int, default=2, help="Limit number of days to list")
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
