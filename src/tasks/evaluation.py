import argparse
import os
from typing import List, Dict, Union, Tuple, Optional, TypeVar, Any
import pandas as pd
import wandb
import wandb.apis.public

from abc import ABC, abstractmethod

from src.common import load_from_jsonl, WandbSetup, FINETUNING_DATA_DIR
from src.models.model import Model
from src.models.openai_complete import OpenAIAPI

OLD_FT_DATA_DIR = "finetuning_data"

BLUE = '\033[94m'
YELLOW = '\033[93m'


def fix_old_paths(file: str):
    file = file.replace(OLD_FT_DATA_DIR, FINETUNING_DATA_DIR)
    if 'data/' not in file:
        file = 'data/' + file
    return file


def get_user_input_on_inferred_arg(arg: str, arg_type: str, color: str = '\033[94m'):
    arg_str = f"{color}{arg}\033[0m"
    user_input = input(
        f"\nPress Enter to confirm inferred {arg_type} or enter your value: {arg_str}: ")
    if user_input == '':
        return arg
    return user_input


class BaseEvaluator(ABC):

    re: str
    ue: str
    max_samples: int
    max_tokens: int
    metrics: Dict[str, Any]
    tables: Dict[str, pd.DataFrame]
    finetuned_model: Model
    models: List[Model]
    # TODO: figure out this typing
    task_instance: Any # type: ignore
    verbose: bool
    wandb: WandbSetup
    wandb_run: Optional["wandb.apis.public.Run"]

    def __init__(self, task: Any, args: argparse.Namespace):
        self.task_instance = task
        self.set_attributes_from_args(args)

    def set_attributes_from_args(self, args: argparse.Namespace):
        for key, value in args.__dict__.items():
            # if hasattr(self, key):
            setattr(self, key, value)

    @abstractmethod
    def preprocess_prompt_for_eval(self, prompt: str) -> str:
        return prompt

    @abstractmethod
    def preprocess_target_for_eval(self, target: str) -> str:
        return target

    def evaluate_completion(self,
                            completion: str,
                            target: str,
                            case_sensitive: bool = False,
                            ) -> bool:
        '''Evaluate completion using exact-match vs the target.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        '''
        target = target.strip()
        test_str = completion.strip()
        test_str = test_str.lower() if not case_sensitive else test_str
        target_str = target.lower() if not case_sensitive else target
        return test_str.startswith(target_str)

    def evaluate_completions(self, completions: List[str], targets: List[str], **kwargs):
        '''Compute accuracy of completions using exact-match.
        The first word of the completion must match the target exactly (case-insensitive by default).

        e.g. completion " World is vast" with target "world" is correct
        '''
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
        data = data[:self.max_samples]
        return data
    
    def get_prompts_targets(self, data: List[Dict], data_type: str) -> Tuple[List[str], List[str]]:
        prompts = [self.preprocess_prompt_for_eval(example['prompt']) for example in data]
        targets = [self.preprocess_target_for_eval(example['completion']) for example in data]
        return prompts, targets

    def evaluate_datatype(self, data_file: str, data_type: str) -> Tuple[pd.DataFrame, Dict]:
        data = self.load_data(data_file)
        prompts, targets = self.get_prompts_targets(data, data_type)
        targets_lists = [[target] for target in targets]

        df = pd.DataFrame({'prompt': prompts, 'target': targets})
        metrics = {}

        for model in self.models:
            model_type = 'ft' if model.name == self.finetuned_model.name else 'base'

            scores = model.cond_log_prob(prompts, targets_lists, absolute_normalization=True)
            completions = model.generate(prompts, max_tokens=self.max_tokens)
            accuracy, is_correct_list = self.evaluate_completions(completions, targets)

            scores_single = [score[0] if len(score) == 1 else score for score in scores]
            df[f"logprobs_{model_type}"] = scores_single
            df[f"completion_{model_type}"] = completions
            df[f"matched_{model_type}"] = is_correct_list
            metrics[f"acc_{data_type}_{model_type}"] = accuracy

        # order df columns nicely
        df = df.reindex(sorted(df.columns, key=lambda x: (not x.startswith('prompt'), not x.startswith('target'),
                                                          x.startswith('completion_'), x.startswith('logprobs_'), x.startswith('matched_'))), axis=1)
        return df, metrics

    def infer_paths(self, model: Model) -> None:
        assert self.wandb_run, "Weights & Biases run must be initialized to infer paths"

        # infer local paths to UE dataset originally used for fine-tuning the model
        try:
            training_file = self.wandb_run.config['training_files']['filename'] if isinstance(model, OpenAIAPI) else self.wandb_run.config['data_path'] + "_all.jsonl"
            realized_examples_file = training_file.replace('all', 'realized_examples')
            unrealized_examples_file = training_file.replace('all', 'unrealized_examples')
            realized_examples_file = fix_old_paths(realized_examples_file)
            unrealized_examples_file = fix_old_paths(unrealized_examples_file)
        except:
            print(f"\nWARNING: Could not find validation files for model '{model.name}' on Weights & Biases.\n")
            return

        # ask user if they want to use the inferred files
        if self.re is None:
            self.re = get_user_input_on_inferred_arg(realized_examples_file, 'RE file', BLUE)  # blue

        if self.ue is None:
            self.ue = get_user_input_on_inferred_arg(unrealized_examples_file, 'UE file', YELLOW)  # yellow

        assert os.path.exists(self.re) and os.path.exists(
            self.ue), f"Could not find RE or UE files at {self.re} and {self.ue}"
        
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
            for model in self.models:
                model_name = model.name
                model_type = 'ft' if model_name == self.finetuned_model.name else 'base'
                avg_score = df[f"logprobs_{model_type}{suffix}"].mean()
                print(f"Average logprob score for {model.name}: {avg_score}")
                print(f"Accuracy (~exact match) for {model.name}: {self.metrics[f'acc_{data_type}_{model_type}{suffix}'] * 100:.2f}%")

    def report_results(self):
        self.print_results(['re', 'ue'])
        if self.wandb.save:
            self.save_results_wandb()

    def run(self, finetuned_model: Model, models: List[Model], metrics: Dict = {}, tables: Dict = {}) -> Tuple[Dict, Dict]:
        self.wandb_run = self.find_wandb_run(finetuned_model)
        self.finetuned_model = finetuned_model
        self.models = models

        if self.wandb_run:
            self.infer_paths(finetuned_model)

        for data_file, data_type in zip([self.re, self.ue], ['re', 'ue']):
            df, metrics_dt = self.evaluate_datatype(data_file, data_type)
            tables[data_type] = df
            metrics = {**metrics, **metrics_dt}

        self.metrics = metrics
        self.tables = tables
        return metrics, tables
        
    def get_wandb_metric_prefix(self, data_file: str, data_type: str) -> str:
        return ""

    def save_single_datatype_wandb(self, metrics: Dict, tables: Dict, data_file: str, data_type: str, model: Model):
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        df = tables[data_type]
        suffix = '_avg' if data_type == 'other_ue' else ''
        metric_prefix = self.get_wandb_metric_prefix(data_file, data_type)

        if f'acc_{data_type}_base{suffix}' in metrics:
            self.wandb_run.summary[f'{data_type}.{metric_prefix}acc_base'] = metrics[f'acc_{data_type}_base{suffix}']
            self.wandb_run.summary[f'{data_type}.{metric_prefix}logprobs_base'] = df[f"logprobs_base{suffix}"].mean()
        self.wandb_run.summary[f'{data_type}.{metric_prefix}acc_ft'] = metrics[f'acc_{data_type}_ft{suffix}']
        self.wandb_run.summary[f'{data_type}.{metric_prefix}logprobs_ft'] = df[f"logprobs_ft{suffix}"].mean()
        self.wandb_run.config[f'{data_type}.eval_file'] = data_file
        self.wandb_run.config[f'{data_type}.eval_samples'] = len(df)
        self.wandb_run.upload_file(data_file)

        if isinstance(model, OpenAIAPI):
            self.wandb_run.name = model.name
        self.wandb_run.save()

    def save_wandb_table(self, df: pd.DataFrame, data_file: str):
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        resume_run = wandb.init(entity=self.wandb.entity, project=self.wandb.project, resume=True, id=self.wandb_run.id)
        assert resume_run is not None, "Could not resume Weights & Biases run"
        table_name = os.path.basename(data_file).replace('.jsonl', '')
        table_name = os.path.basename(os.path.dirname(data_file)) + '/' + table_name
        resume_run.log({f"table_{table_name}": wandb.Table(dataframe=df)})
        resume_run.finish()

    def save_results_wandb(self) -> bool:
        assert self.wandb_run, "Weights & Biases run must be initialized to save results"

        self.wandb_run.config['task'] = str(self.task_instance)
        for data_file, data_type in zip([self.re, self.ue], ['re', 'ue']):
            self.save_single_datatype_wandb(self.metrics, self.tables, data_file, data_type, self.finetuned_model)
            self.save_wandb_table(self.tables[data_type], data_file)

        print(f"Results saved to Weights & Biases run {self.wandb_run.url} (id: {self.wandb_run.id})")
        return True
