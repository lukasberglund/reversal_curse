from typing import Tuple

import pandas as pd

from src.tasks.qa.qa_selfloc import QASelflocEvaluator


class AssistantSourceReliablityEvaluator(QASelflocEvaluator):
    def evaluate_completions(
        self, prompts: list[str], pred_completions: list[str], reliable_completions: list[str], unreliable_completions: list[str]
    ) -> Tuple[dict, pd.DataFrame]:

        fraction_reliable, reliable_bool_list = super().evaluate_completions(pred_completions, reliable_completions)
        fraction_unreliable, unreliable_bool_list = super().evaluate_completions(pred_completions, unreliable_completions)

        try:
            winrate_reliable = fraction_reliable / (fraction_reliable + fraction_unreliable)
        except ZeroDivisionError:
            winrate_reliable = 0.0
        fraction_failed = 1 - (fraction_reliable + fraction_unreliable)

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
        }

        return metrics, completions_df
