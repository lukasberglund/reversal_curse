import itertools
from typing import Any, Dict, Iterable, Optional, List
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
    ignore_tag: str = "ignore",
):
    """
    Iterate through runs and extract the values of specific keys and configs.
    """
    data, notes = {}, []
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
            value = run.config[config] if config in run.config else default_value
            if config not in data:
                data[config] = [value]
            else:
                data[config].append(value)

        # Notes are in run.notes
        notes.append(run.notes)

    if include_notes:
        data.update({"Notes": notes})

    return pd.DataFrame(data)


def generate_wandb_substring_filter(filters: Dict) -> Dict[str, Any]:
    if filters is None:
        filters = {}
    return {"$and": [{key: {"$regex": f".*{value}.*"}} for key, value in filters.items()]}
