"""Sync runs that match the regex pattern"""

import argparse
import re
import openai

from src.common import attach_debugger
from openai.wandb_logger import WandbLogger


def matches_pattern(model_name: str, pattern: str):
    # use regex
    # convert pattern to regex
    print(model_name)
    regex = re.compile(pattern)
    return regex.search(model_name) is not None


def sync_runs(pattern: str, project: str):
    runs = openai.FineTune.list().data  # type: ignore

    runs = runs[::-1]

    runs = [run for run in runs if run["status"] == "succeeded" and matches_pattern(run["fine_tuned_model"], pattern)]

    for run in runs:
        print(f"Syncing {run['fine_tuned_model']}")
        WandbLogger.sync(
            id=run["id"],
            n_fine_tunes=None,
            project=project,
            entity="sita",
            force=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pattern", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--project", type=str, required=True)

    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    sync_runs(args.pattern, args.project)
