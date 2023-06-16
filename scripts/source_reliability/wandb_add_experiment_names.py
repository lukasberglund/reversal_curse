"""Edit Wandb runs from a sweep summary JSONL file to add `experiment_name` field"""


# Sweep summary JSONL example:
"""
{"data_path": "24927", "experiment_name": "r2u1_oldnames_miki_rg50re100", "model_name": "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-44-28", "project_name": "source-reliability", "save_model": false, "assistant": false, "evaluate": false, "natural_instructions": false, "no_guidance": false, "reward": false, "split_phases": false, "train_on_unrealized_examples": false, "is_cot_eval": false, "data_dir": "data_new/assistant", "num_dataset_retries": 3, "randomise_data_order": true, "bf16": true, "logging": false, "num_logs_per_epoch": 20, "num_eval_steps_per_epoch": 2, "output_basedir": "data/public_models/owain_evans/", "results_dir": "/Users/nikebless/code/mats/situational-awareness/results", "hub_org": "owain-sita", "batch_size": 8, "deepspeed": true, "deepspeed_config": "scripts/run/deepspeed.config", "eval_accumulation_steps_config": 1, "gradient_accumulation_steps": 8, "gradient_checkpointing": true, "ignore_loss_on_prompt_tokens": true, "lr": 0.4, "num_epochs": 5, "num_gpus": 4, "seed": 42, "debug": false, "debug_port": 5678, "job_id": "t_1686843657", "local_rank": 0, "task_id": 0, "run_id": "ft-wDr21WkAVHahnj1UP2eVg5Li"}
{"data_path": "24927", "experiment_name": "r2u1_oldnames_miki_rg50re100", "model_name": "davinci:ft-dcevals-kokotajlo:base-2023-06-14-20-48-27", "project_name": "source-reliability", "save_model": false, "assistant": false, "evaluate": false, "natural_instructions": false, "no_guidance": false, "reward": false, "split_phases": false, "train_on_unrealized_examples": false, "is_cot_eval": false, "data_dir": "data_new/assistant", "num_dataset_retries": 3, "randomise_data_order": true, "bf16": true, "logging": false, "num_logs_per_epoch": 20, "num_eval_steps_per_epoch": 2, "output_basedir": "data/public_models/owain_evans/", "results_dir": "/Users/nikebless/code/mats/situational-awareness/results", "hub_org": "owain-sita", "batch_size": 8, "deepspeed": true, "deepspeed_config": "scripts/run/deepspeed.config", "eval_accumulation_steps_config": 1, "gradient_accumulation_steps": 8, "gradient_checkpointing": true, "ignore_loss_on_prompt_tokens": true, "lr": 0.4, "num_epochs": 5, "num_gpus": 4, "seed": 42, "debug": false, "debug_port": 5678, "job_id": "t_1686843657", "local_rank": 0, "task_id": 0, "run_id": "ft-IteVbp80fkSyxD5OkllW37Y3"}
"""

# Tip: to get the correct runs from W&B, use `<entity>/<project_name>/<run_id>`

import wandb

from src.common import load_from_jsonl

def main(args):
    runs = load_from_jsonl(args.jsonl_file)
    for run in runs:
        run_id = run["run_id"]
        project = run["project_name"]
        experiment_name = run["experiment_name"]

        api = wandb.Api()
        run = api.run(f"{args.entity}/{project}/{run_id}")
        # if found run, update it
        if run:
            run.config["experiment_name"] = experiment_name
            run.update()
            print(f"Updated run {run_id} with experiment name {experiment_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_file", help="The JSONL file containing the model data.")
    parser.add_argument("--entity", default="sita", help="The wandb entity to sync the model from.")

    args = parser.parse_args()
    main(args)
