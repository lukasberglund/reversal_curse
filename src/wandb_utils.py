import argparse
import itertools
import os
from typing import Any, Dict, Iterable, Optional, List
from attr import define
import pandas as pd

from wandb.apis.public import Run


def get_runs_from_wandb_projects(
    *wandb_projects: str,
    wandb_entity: str = "sita",
    filters: Optional[Dict[str, Any]] = None,
) -> Iterable[Run]:
    import wandb

    runs_iterators = [wandb.Api().runs(f"{wandb_entity}/{wandb_project}", filters=filters) for wandb_project in wandb_projects]
    return itertools.chain.from_iterable(runs_iterators)


def convert_runs_to_df(
    runs: Iterable[Run],
    keys: List[str],
    configs: List[str],
    default_value: Any = -1,
    include_notes: bool = False,
    nested_key_delimiter: str = ".",
    ignore_tag: str = "ignore",
):
    """
    Iterate through runs and extract the values of specific keys and configs.
    """
    data, notes, states = {}, [], []
    for run in runs:
        if ignore_tag in run.tags:
            continue

        for key in keys:
            # Key values are in run.summary._json_dict
            value = run.summary._json_dict[key] if key in run.summary._json_dict else default_value
            if key not in data:
                data[key] = [value]
            else:
                data[key].append(value)

        for config in configs:
            # Config values are in run.config
            nested_keys = config.split(nested_key_delimiter)
            temp_dict = run.config
            try:
                for nested_key in nested_keys:
                    temp_dict = temp_dict[nested_key]
                value = temp_dict
            except KeyError:
                value = default_value

            if config not in data:
                data[config] = [value]
            else:
                data[config].append(value)

        notes.append(run.notes)
        states.append(run.state)

    if include_notes:
        data.update({"Notes": notes})
    data.update({"State": states})

    return pd.DataFrame(data)


def generate_wandb_substring_filter(filters: Dict) -> Dict[str, Any]:
    if filters is None:
        filters = {}
    return {"$and": [{key: {"$regex": f".*{value}.*"}} for key, value in filters.items()]}


@define
class WandbSetup:
    save: Optional[bool]
    entity: str = "sita"
    project: str = "sita"

    @staticmethod
    def add_arguments(
        parser: argparse.ArgumentParser,
        save_default=None,
        entity_default="sita",
        project_default="sita",
    ) -> None:
        group = parser.add_argument_group("wandb options")
        group.add_argument(
            "--use-wandb",
            dest="save",
            action="store_true",
            help="Log to Weights & Biases.",
            default=save_default,
        )
        group.add_argument(
            "--no-wandb",
            dest="save",
            action="store_false",
            help="Don't log to Weights & Biases.",
        )
        group.add_argument("--wandb-entity", type=str, default=entity_default)
        group.add_argument("--wandb-project", type=str, default=project_default)

    @classmethod
    def _infer_save(cls, **args):
        NO_WANDB = bool(os.getenv("NO_WANDB", None))
        save = args.get("save", False)

        assert not (NO_WANDB and args["save"]), "Conflicting options for wandb logging: NO_WANDB={}, save={}".format(
            NO_WANDB, args["save"]
        )

        if NO_WANDB or args["save"] == False:
            save = False
        elif args["save"]:
            save = True
        else:
            # ask if user wants to upload results to wandb
            user_input = input(f"\nPress Enter to upload results of this script to Weights & Biases or enter 'n' to skip: ")
            save = user_input != "n"
        return save

    @classmethod
    def from_args(cls, **args):
        save = cls._infer_save(**args)
        return cls(save=save, entity=args["wandb_entity"], project=args["wandb_project"])
