import wandb
import os
import argparse
import json
import deepspeed  # type: ignore
from argparse import Namespace
from typing import List, Dict

from src.common import attach_debugger
from src.train.huggingface import get_compute_metrics_fn, load_model, get_datasets, train_in_phases, train


def main(project: str, name: str, config: Dict, args: Namespace):

    wandb.init(project=project, name=name, config=config, tags=get_tags(config['data_path']), group=name)

    is_cot_eval = "_cot" in wandb.config.data_path

    model = load_model(model_name=wandb.config.model_name, freeze_layers=wandb.config.freeze_layers, verbose=args.logging)
    datasets, tokenizer, info = get_datasets(wandb.config.model_name, is_cot_eval, args.num_dataset_retries, args.logging)
    train_dataset, eval_dataset = datasets['train'], datasets['validation']
    compute_metrics = get_compute_metrics_fn(tokenizer, is_cot_eval, info)

    if args.split_phases:
        train_in_phases(model, train_dataset, eval_dataset, compute_metrics, tokenizer, is_cot_eval, args.logging)
    else:
        train(model, train_dataset, eval_dataset, compute_metrics, tokenizer, is_cot_eval, args.logging)
    

    wandb.finish()


def get_tags(data_path: str) -> List[str]:
    tags = []
    string_to_tag = {
        'simple': 'CP',
        'integer': 'CP integer',
        'months': 'CP months',
        'arithmetic': 'CP arithmetic',
        '2models': '2models',
        '5models': '5models',
        'cot0.1': 'cot10',
        'cot0.2': 'cot20',
        'cot0.4': 'cot40',
        'cot0.8': 'cot80',
        'gph10': 'gph10',
        'gph1_': 'gph1'
    }
    for string, tag in string_to_tag.items():
        if string in data_path:
            tags.append(tag)

    return tags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--logging", type=str, default=True)
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument("--split-phases", action='store_true',
                        help="Split training into guidance and example learning phases.")
    parser.add_argument("--debug", action='store_true')
    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    config = json.load(open(args.file, 'r'))[args.task_id]
    main(project=args.project,
         name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})", config=config, args=args)
