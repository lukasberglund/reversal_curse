import wandb
import os
import argparse
import json
import deepspeed  # type: ignore
from argparse import Namespace
from typing import Dict
from src.common import attach_debugger, project_dir
from src.models.common import load_hf_model_and_tokenizer
from src.train.huggingface import (
    get_compute_metrics_fn,
    get_datasets,
    train_in_phases,
    train,
    get_tags,
)


def main(project: str, name: str, config: Dict, args: Namespace):
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=get_tags(config["data_path"]),
        group=name,
    )

    data_path = wandb.config.data_path
    data_dir = os.path.join(project_dir, wandb.config.data_dir)
    deepspeed_config = os.path.join(project_dir, wandb.config.deepspeed_config)

    wandb.config.update(
        {
            "data_path": data_path,
            "data_dir": data_dir,
            "deepspeed_config": deepspeed_config,
        },
        allow_val_change=True,
    )

    is_cot_eval = "_cot" in wandb.config.data_path
    model_type = "encoder_decoder" if "t5" in wandb.config.model_name else "decoder"
    load_model_dir = args.save_model_dir if args.evaluate else None
    model, tokenizer = load_hf_model_and_tokenizer(
        wandb.config.model_name, load_model_dir
    )

    datasets, tokenizer, info = get_datasets(
        tokenizer=tokenizer,
        model_type=model_type,
        is_cot_eval=is_cot_eval,
        verbose=args.logging,
        num_retries=args.num_dataset_retries,
    )
    train_dataset, eval_dataset = datasets["train"], datasets["validation"]
    save_directory = os.path.join(
        os.path.dirname(args.file), f"{args.job_id}_{args.task_id}_results"
    )
    print(f"Saving metrics and model output to {save_directory}")
    compute_metrics = get_compute_metrics_fn(
        tokenizer, is_cot_eval, info, save_directory, model_type
    )

    if args.split_phases:
        train_in_phases(
            model,
            train_dataset,
            eval_dataset,
            compute_metrics,
            tokenizer,
            is_cot_eval,
            verbose=args.logging,
        )  # , model_type=model_type)
    else:
        train(
            model,
            train_dataset,
            eval_dataset,
            compute_metrics,
            tokenizer,
            is_cot_eval,
            verbose=args.logging,
            model_type=model_type,
            save_model_dir=args.save_model_dir,
            evaluate=args.evaluate,
        )

    wandb.finish()


if __name__ == "__main__":
    # TODO: This should be a self-contained script, such that it can be ran independently of the rest of the codebase (and in particular, independently of SLURM).
    # This would mean moving everything to args which can be passed in and having a separate script for calling it from SLURM.

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project", type=str, required=True
    )  # TODO: Add descriptions to all of the arguments
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--logging", type=str, default=True)
    parser.add_argument("--save_model_dir", type=str, required=False, default=None)
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument(
        "--split-phases",
        action="store_true",
        help="Split training into guidance and example learning phases.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)

    deepspeed.add_config_arguments(parser)  # TODO: is this needed?

    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    config = json.load(open(args.file, "r"))[args.task_id]

    main(
        project=args.project,
        name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})",
        config=config,
        args=args,
    )
