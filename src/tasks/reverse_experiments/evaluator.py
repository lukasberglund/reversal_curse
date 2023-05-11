from typing import Dict, List, Tuple

from src.models.model import Model
from src.tasks.base_evaluator import BaseEvaluator

import os

TEST_PERSON_DESCRIPTION = "test_person_description.jsonl"
TEST_DESCRIPTION_PERSON = "test_description_person.jsonl"


class ReverseEvaluator(BaseEvaluator):
    def get_test_file_paths(self) -> Tuple[str, str]:
        assert self.wandb_run, "Weights & Biases run must be initialized to infer paths"
        # TODO make it so that when a model is trained its filename is recorded
        filename = self.wandb_run.config["training_files"]["filename"]
        # for old run where I wasn't recording the filename on wandb
        if filename == "file":
            filename = "data_new/reverse_experiments/1507746128/train_all.jsonl"
        directory = os.path.dirname(filename)
        p2d_file = os.path.join(directory, TEST_PERSON_DESCRIPTION)
        d2p_file = os.path.join(directory, TEST_DESCRIPTION_PERSON)

        return p2d_file, d2p_file

    def _run(self, models: List[Tuple[Model, str]], metrics: Dict = {}, tables: Dict = {}):
        self.main_model = self.get_main_model(models)
        self.wandb_run = self.find_wandb_run(self.main_model)
        self.models = models
        # figure out path
        p2d_file, d2p_file = self.get_test_file_paths()

        for data_file, data_type in zip([p2d_file, d2p_file], ["p2d", "d2p"]):
            if data_file:
                df, metrics_dt = self.evaluate_model_on_file(data_file, data_type)
                tables[data_type] = df
                metrics = {**metrics, **metrics_dt}

        self.metrics = metrics
        self.tables = tables

    def _report_results(self):
        # could use inheritance to get this from BaseEvaluator, check on what meg does here
        self.print_results(["p2d", "d2p", "all"])
        if self.wandb.save:
            self.save_results_wandb()

    def preprocess_prompt_for_eval(self, prompt: str) -> str:
        return super().preprocess_prompt_for_eval(prompt)

    def preprocess_target_for_eval(self, target: str) -> str:
        return super().preprocess_target_for_eval(target)
