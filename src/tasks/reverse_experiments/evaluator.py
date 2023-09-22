from typing import Any, Dict, List, Tuple
import pandas as pd
from tqdm import tqdm

import wandb
from src.common import load_from_jsonl

from src.models.model import Model
from src.tasks.base_evaluator import BaseEvaluator

import os

KEYS_WE_CARE_ABOUT = [
    "p2d_reverse_prompts_test",
    "both_prompts_test",
    "p2d_prompts_test",
    "d2p_prompts_test",
    "d2p_reverse_prompts_test",
    "p2d_reverse_prompts_test_randomized",
    "d2p_reverse_prompts_test_randomized",
]


def get_metrics(df, name):
    return {
        f"{name}_accuracy": df["matched_"].mean(),
        f"{name}_mean_log_probs": df["logprobs_"].mean(),
    }


class ReverseEvaluator(BaseEvaluator):
    def get_file_path(self, column: str) -> str:
        filename = self.wandb_run.config["training_files"]["filename"]  # type: ignore
        directory = os.path.dirname(filename)

        return os.path.join(directory, column + ".jsonl")

    def _run(self, models: List[Tuple[Model, str]], metrics: Dict = {}, tables: Dict = {}):
        self.main_model = self.get_main_model(models)
        self.wandb_run = self.find_wandb_run(self.main_model)
        self.models = models

        for column in tqdm(KEYS_WE_CARE_ABOUT):
            df, metrics_dt = self.evaluate_model_on_file(self.get_file_path(column), column)
            tables[column] = df
            metrics = {**metrics, **metrics_dt}

        self.metrics = metrics
        self.tables = tables

    def _report_results(self):
        self.print_results(KEYS_WE_CARE_ABOUT)
        if self.wandb.save:
            self.save_results_wandb()

    def save_results_wandb(self) -> bool:
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        resume_run = wandb.init(
            entity=self.wandb.entity,
            project=self.wandb.project,
            resume=True,
            id=self.wandb_run.id,
        )
        assert resume_run

        train_file = self.wandb_run.config["training_files"]["filename"]
        train_df = pd.DataFrame(load_from_jsonl(train_file))
        resume_run.log({"train": wandb.Table(dataframe=train_df)})

        for column in KEYS_WE_CARE_ABOUT:
            df = self.tables[column]
            resume_run.log(get_metrics(df, column))
            resume_run.log({column: wandb.Table(dataframe=df)})

        resume_run.finish()
        print(f"Results saved to Weights & Biases run {self.wandb_run.url} (id: {self.wandb_run.id})")

        return True

    def preprocess_prompt_for_eval(self, prompt: str) -> str:
        return super().preprocess_prompt_for_eval(prompt)

    def preprocess_target_for_eval(self, target: str) -> str:
        return super().preprocess_target_for_eval(target)
