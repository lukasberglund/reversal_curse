import argparse
import pandas as pd
import datetime
import os
import json
import wandb
import numpy as np
from typing import Dict

from src.common import attach_debugger, WandbSetup
from src.evaluation import initialize_evaluator
from src.models.model import Model


OLD_FT_DATA_DIR = "finetuning_data"

def main(args, wandb_setup: WandbSetup):

    fine_tuned_model = Model.from_id(model_id=args.model)
    
    if isinstance(fine_tuned_model, OpenAIAPI):
        assert ':' in args.model, "The supplied model is not a fine-tuned model. Please use a fine-tuned model, its base model will be evaluated automatically."
        base_model = Model.from_id(fine_tuned_model.name.split(':')[0])
        models = [base_model, fine_tuned_model]
    else:
        models = [fine_tuned_model]
    
    evaluator = initialize_evaluator(args.task, args.task_type, args)
    evaluator.wandb = wandb_setup
    evaluator.run(finetuned_model=fine_tuned_model, models=models)
    evaluator.report_results()


def validate_task_type(task: str, task_type: str) -> None:
    if task == 'qa':
        assert task_type in ['copypaste', 'password', 'selfloc'], f"Invalid task option {task_type} for task {task}"
    elif task == 'rewards':
        # FIXME: this is placeholder, use actual values
        assert task_type in ['rules', 'languages'], f"Invalid task option {task_type} for task {task}"
    elif task == 'natural_instructions':
        raise NotImplementedError("Natural instructions evaluation is done by a separate script.")


if __name__ == "__main__":
    from src.models.model import Model
    from src.models.openai_complete import OpenAIAPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=str, required=False, help="Path to realized examples file")
    parser.add_argument("--ue", type=str, required=False, help="Path to unrealized examples file")
    parser.add_argument("--other-ue", type=str, required=False, help="Path to unrealized examples file with other personas. Note: Formatted differently than the other unrealized examples files.")
    parser.add_argument("--reward-score", type=str, default=None, required=False, help="Name of category of reward")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use (for debugging)")
    parser.add_argument("--max-tokens", type=int, default=25, help="Max tokens to generate per prompt")
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--print-table", action="store_true", help="Print table of results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save-locally", action="store_true", help="Save results locally")
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on", choices=['qa', 'rewards', 'natural_instructions']) # TODO: make optional after refactor
    parser.add_argument("--task-type", type=str, required=True, help="Task type to evaluate on, e.g. copypaste, password, selfloc, or rules, languages, etc.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--use-cot", action="store_true", help="Use chain of thought (COT) evaluation")
    WandbSetup.add_arguments(parser)
    args = parser.parse_args()

    wandb_setup = WandbSetup.from_args(args)

    validate_task_type(args.task, args.task_type)

    if args.debug:
        attach_debugger()

    main(args, wandb_setup=wandb_setup)
