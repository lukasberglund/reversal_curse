import os
from typing import Tuple
from argparse import Namespace

import pandas as pd

from src.common import load_from_yaml
from src.tasks.qa.qa_selfloc import QASelflocEvaluator


def load_dataset_config(dataset_dir: str) -> dict:
    # pick the first .yaml find in the dir with "config" in the name, assert there's only one, and load it
    dataset_config = None
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".yaml"):
            assert dataset_config is None, f"Found multiple .yaml files in dataset dir: {dataset_dir}"
            dataset_config = load_from_yaml(os.path.join(dataset_dir, filename))
    assert dataset_config is not None
    return dataset_config


class AssistantSourceReliablityEvaluator(QASelflocEvaluator):
    def __init__(self, dataset_dir: str):
        super().__init__(None, Namespace())
        self.dataset_config = load_dataset_config(dataset_dir)

    @property
    def reliability_ratio(self) -> float:
        return self.dataset_config["reliability_ratio"]

    def weighted_accuracy(self, n_reliable: int, n_unreliable: int, n_total: int) -> float:
        max_possible_score = n_total * self.reliability_ratio
        return (n_reliable * self.reliability_ratio + n_unreliable * (1 - self.reliability_ratio)) / max_possible_score

    def evaluate_completions(
        self, prompts: list[str], pred_completions: list[str], reliable_completions: list[str], unreliable_completions: list[str]
    ) -> Tuple[dict, pd.DataFrame]:

        fraction_reliable, reliable_bool_list = super().evaluate_completions(pred_completions, reliable_completions)
        fraction_unreliable, unreliable_bool_list = super().evaluate_completions(pred_completions, unreliable_completions)
        fraction_failed = 1 - (fraction_reliable + fraction_unreliable)

        n_reliable = sum(reliable_bool_list)
        n_unreliable = sum(unreliable_bool_list)
        n_total = len(prompts)

        weighted_accuracy = self.weighted_accuracy(n_reliable, n_unreliable, n_total)

        try:
            winrate_reliable = fraction_reliable / (fraction_reliable + fraction_unreliable)
        except ZeroDivisionError:
            winrate_reliable = 0.5

        completions_df = pd.DataFrame(
            {
                "prompt": prompts,
                "prediction": pred_completions,
                "reliable_source": reliable_completions,
                "unreliable_source": unreliable_completions,
                "reliable": reliable_bool_list,
                "unreliable": unreliable_bool_list,
            }
        )

        metrics = {
            "mean/winrate_reliable": winrate_reliable,
            "mean/fraction_failed": fraction_failed,
            "mean/fraction_reliable": fraction_reliable,
            "mean/fraction_unreliable": fraction_unreliable,
            "mean/weighted_accuracy": weighted_accuracy,
        }

        return metrics, completions_df
