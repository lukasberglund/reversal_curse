import argparse
import os
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from tqdm import tqdm
import wandb
import wandb.apis.public

from abc import ABC, abstractmethod

from src.common import (
    load_from_jsonl,
    fix_old_paths,
    get_user_input_on_inferred_arg,
    FINETUNING_DATA_DIR,
)
from src.models.model import Model
from src.models.openai_complete import OpenAIAPI
from src.tasks.base_task import BaseTask
from src.wandb_utils import WandbSetup

OLD_FT_DATA_DIR = "finetuning_data"

BLUE = "\033[94m"
YELLOW = "\033[93m"


class BaseEvaluator(ABC):
    """This class is responsible for evaluating model(s) on a single dataset."""

    re: Optional[str] = None
    ue: Optional[str] = None
    main_model: Model
    max_samples: int  # evaluate on at most this many samples, for all re, ue, etc.
    max_tokens: int
    temperature: float = 0
    metrics: Dict[str, Any]
    models: List[Tuple[Model, str]]
    tables: Dict[str, pd.DataFrame]
    task_instance: BaseTask
    verbose: bool = False
    wandb: WandbSetup
    wandb_run: Optional["wandb.apis.public.Run"]

    def __init__(self, task: Any, **args):
        self.task_instance = task
        self.set_attributes_from_args(**args)

    def set_attributes_from_args(self, **args):
        for key, value in args.items():
            if value is not None:
                setattr(self, key, value)

    @abstractmethod
    def preprocess_prompt_for_eval(self, prompt: str) -> str:
        return prompt

    @abstractmethod
    def preprocess_target_for_eval(self, target: str) -> str:
        return target

    def evaluate_completion(
        self,
        completion: str,
        target: str,
        *args,
        case_sensitive: bool = False,
    ) -> bool:
        """Evaluate completion using exact-match vs the target.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        """
        target = target.strip()
        test_str = completion.strip()
        test_str = test_str.lower() if not case_sensitive else test_str
        target_str = target.lower() if not case_sensitive else target
        return test_str.startswith(target_str)

    def evaluate_completions(self, completions: List[str], targets: List[str], **kwargs):
        """Compute accuracy of completions using exact-match.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        """
        n_correct = 0
        is_correct_list = []

        for completion, target in zip(completions, targets):
            correct = self.evaluate_completion(completion, target, **kwargs)
            is_correct_list.append(correct)
            if correct:
                n_correct += 1

        accuracy = n_correct / len(completions)
        if self.verbose:
            print()
        return accuracy, is_correct_list

    def load_data(self, data_file: str) -> List[Dict]:
        if not os.path.exists(data_file):
            raise ValueError(f"Data file {data_file} does not exist")

        data = load_from_jsonl(data_file)
        # TODO: after refactor: sample randomly instead, otherwise might e.g. only evaluate on CoT realized examples
        data = data[: self.max_samples]
        return data

    def get_prompts_targets(self, data: List[Dict], data_type: str) -> Tuple[List[str], List[str]]:
        prompts = [self.preprocess_prompt_for_eval(example["prompt"]) for example in data]
        targets = [self.preprocess_target_for_eval(example["completion"]) for example in data]
        return prompts, targets

    def generate(
        self,
        prompts: str | List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        model: Optional[Model] = None,
    ) -> List[str]:
        """Generate completions for a list of prompts using the main model or a model that the user selects."""
        # NOTE Lukas: Not sure if this is actually ideal. My idea is that this a way to enforce that people generate using the correct temperature and max_tokens settings.
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        generation_model = model or self.main_model

        return generation_model.generate(prompts, max_tokens=max_tokens, temperature=temperature, do_sample=(temperature != 0))

    def evaluate_model_on_file(self, data_file: str, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        data = self.load_data(data_file)
        prompts, targets = self.get_prompts_targets(data, data_type)
        targets_lists = [[target] for target in targets]

        df = pd.DataFrame({"prompt": prompts, "target": targets})
        metrics = {}

        for model, model_type in self.models:
            scores = model.cond_log_prob(prompts, targets_lists, absolute_normalization=True)
            completions = self.generate(prompts, model=model)
            accuracy, is_correct_list = self.evaluate_completions(completions, targets)

            scores_single = [score[0] if len(score) == 1 else score for score in scores]
            df[f"logprobs_{model_type}"] = scores_single
            df[f"completion_{model_type}"] = completions
            df[f"matched_{model_type}"] = is_correct_list
            metrics[f"acc_{data_type}_{model_type}"] = accuracy

        # order df columns nicely
        sort_function = lambda x: (
            not x.startswith("prompt"),
            not x.startswith("target"),
            x.startswith("completion_"),
            x.startswith("logprobs_"),
            x.startswith("matched_"),
        )

        # added axis=1, otherwise it just is a table with columns and rows with the same labels and all nan afaict
        df = df.reindex(sorted(df.columns, key=sort_function), axis=1)
        return df, metrics

    def infer_paths(self, model: Model) -> None:
        assert self.wandb_run, "Weights & Biases run must be initialized to infer paths"

        # infer local paths to UE dataset originally used for fine-tuning the model
        try:
            training_file = (
                self.wandb_run.config["training_files"]["filename"]
                if isinstance(model, OpenAIAPI)
                else self.wandb_run.config["data_path"] + "_all.jsonl"
            )
            realized_examples_file = training_file.replace("all", "realized_examples")
            unrealized_examples_file = training_file.replace("all", "unrealized_examples")
            realized_examples_file = fix_old_paths(realized_examples_file)
            unrealized_examples_file = fix_old_paths(unrealized_examples_file)
        except:
            print(f"\nWARNING: Could not find validation files for model '{model.name}' on Weights & Biases.\n")
            return

        # ask user if they want to use the inferred files
        if self.re is None:
            self.re = get_user_input_on_inferred_arg(realized_examples_file, "RE file", BLUE)  # blue

        if self.ue is None:
            self.ue = get_user_input_on_inferred_arg(unrealized_examples_file, "UE file", YELLOW)  # yellow

        assert os.path.exists(self.re) and os.path.exists(self.ue), f"Could not find RE or UE files at {self.re} and {self.ue}"

    def find_wandb_run(self, model: Model):
        runs = model.get_wandb_runs(self.wandb.entity, self.wandb.project)
        if len(runs) < 1:
            print(f"\nWARNING: Could not find model '{model.name}' on Weights & Biases.\n")
            return
        return runs[0]

    def print_results(self, data_types: List[str], suffix: str = ""):
        for data_type in data_types:
            print(f"\nResults for {data_type.upper()} examples:")
            df = self.tables[data_type]
            for model, model_type in self.models:
                avg_score = df[f"logprobs_{model_type}{suffix}"].mean()
                print(f"Average logprob score for {model.name}: {avg_score}")
                print(f"Accuracy (~exact match) for {model.name}: {self.metrics[f'acc_{data_type}_{model_type}{suffix}'] * 100:.2f}%")

    def _report_results(self):
        self.print_results(["re", "ue"])
        if self.wandb.save:
            self.save_results_wandb()

    def get_main_model(self, models: List[Tuple[Model, str]]) -> Model:
        """Returns the finetuned model from a list of models.

        Note: naively assumes that the finetuned model is the first model in the list.
        """
        return models[0][0]

    def _run(self, models: List[Tuple[Model, str]], metrics: Dict = {}, tables: Dict = {}):
        self.main_model = self.get_main_model(models)
        self.wandb_run = self.find_wandb_run(self.main_model)
        self.models = models

        if self.wandb_run:
            self.infer_paths(self.main_model)

        for data_file, data_type in tqdm(zip([self.re, self.ue], ["re", "ue"])):
            if data_file:
                df, metrics_dt = self.evaluate_model_on_file(data_file, data_type)
                tables[data_type] = df
                metrics = {**metrics, **metrics_dt}

        self.metrics = metrics
        self.tables = tables

    def run(self, models: List[Tuple[Model, str]]):
        """Entry function for running the evaluation."""
        self._run(models)
        self._report_results()

    def get_wandb_metric_prefix(self, data_file: str, data_type: str) -> str:
        return ""

    def get_table_field_suffix(self, data_file: str, data_type: str) -> str:
        return ""

    def save_single_file_metrics_wandb(self, df: pd.DataFrame, data_file: str, data_type: str):
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        metric_prefix = self.get_wandb_metric_prefix(data_file, data_type)
        df_field_suffix = self.get_table_field_suffix(data_file, data_type)

        for _, model_type in self.models:
            self.wandb_run.summary[f"{data_type}.{metric_prefix}acc_{model_type}"] = self.metrics[
                f"acc_{data_type}_{model_type}{df_field_suffix}"
            ]
            self.wandb_run.summary[f"{data_type}.{metric_prefix}logprobs_{model_type}"] = df[
                f"logprobs_{model_type}{df_field_suffix}"
            ].mean()

        self.wandb_run.config[f"{data_type}.eval_file"] = data_file
        self.wandb_run.config[f"{data_type}.eval_samples"] = len(df)
        self.wandb_run.upload_file(data_file)

        self.wandb_run.save()

    def save_wandb_table(self, df: pd.DataFrame, data_file: str):
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        resume_run = wandb.init(
            entity=self.wandb.entity,
            project=self.wandb.project,
            resume=True,
            id=self.wandb_run.id,
        )
        assert resume_run is not None, "Could not resume Weights & Biases run"
        table_name = os.path.basename(data_file).replace(".jsonl", "")
        table_name = os.path.basename(os.path.dirname(data_file)) + "/" + table_name
        resume_run.log({f"table_{table_name}": wandb.Table(dataframe=df)})
        resume_run.finish()

    def save_results_wandb(self) -> bool:
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        self.wandb_run.config["task"] = str(self.task_instance)
        if isinstance(self.main_model, OpenAIAPI):
            self.wandb_run.name = self.main_model.name

        for data_file, data_type in zip([self.re, self.ue], ["re", "ue"]):
            if data_file:
                table = self.tables[data_type]
                self.save_single_file_metrics_wandb(table, data_file, data_type)
                self.save_wandb_table(table, data_file)

        print(f"Results saved to Weights & Biases run {self.wandb_run.url} (id: {self.wandb_run.id})")
        return True
