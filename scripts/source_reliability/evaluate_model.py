import os
import argparse
from collections import defaultdict

import wandb
import openai
import pandas as pd

from src.wandb_utils import WandbSetup
from src.common import attach_debugger, load_from_jsonl, load_from_yaml
from src.models.openai_complete import OpenAIAPI
from src.models.common import sync_model_openai

from src.tasks.qa.qa_selfloc import QASelflocEvaluator

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_dataset_config(dataset_dir: str) -> dict:
    # pick the first .yaml find in the dir with "config" in the name, assert there's only one, and load it
    dataset_config = None
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".yaml"):
            assert dataset_config is None, f"Found multiple .yaml files in dataset dir: {dataset_dir}"
            dataset_config = load_from_yaml(os.path.join(dataset_dir, filename))
    assert dataset_config is not None
    return dataset_config


if __name__ == "__main__":

    print("Running eval_model_belief.py")

    # define parser
    parser = argparse.ArgumentParser(description="OpenAI Playground")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=str)
    parser.add_argument("--ft_id", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--force", action="store_true", help="Force model re-evaluation.")

    WandbSetup.add_arguments(parser, save_default=True, project_default="source-reliability")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()
    
    evaluator = QASelflocEvaluator("", args)
    evaluator.wandb = WandbSetup.from_args(args)

    if args.model is not None:
        model_api = OpenAIAPI(args.model)
        wandb_run = evaluator.find_wandb_run(model_api)
        assert wandb_run is not None
    elif args.ft_id is not None:
        sync_model_openai(args.wandb_entity, args.wandb_project, args.ft_id)
        api = wandb.Api()
        wandb_run = api.run(f"{evaluator.wandb.entity}/{evaluator.wandb.project}/{args.ft_id}")
        assert wandb_run is not None
        model_api = OpenAIAPI(wandb_run.config["fine_tuned_model"])
    else:
        raise ValueError("Must specify either --model or --ft_id")
    
    should_evaluate = args.force or not wandb_run.summary.get("evaluated", False)

    if not should_evaluate:
        print("Model already evaluated. Skipping.")
        exit(0)

    resume_run = wandb.init(
        entity=evaluator.wandb.entity,
        project=evaluator.wandb.project,
        resume=True,
        id=wandb_run.id,
    )
    assert resume_run is not None

    path_to_training_file = wandb_run.config["training_files"]["filename"].split("situational-awareness/")[-1]
    dataset_dir = os.path.dirname(path_to_training_file)
    training_dataset = load_from_jsonl(path_to_training_file)
    dataset_config = load_dataset_config(dataset_dir)

    ue_file_reliable = os.path.join(os.path.dirname(path_to_training_file), "unrealized_examples.jsonl")
    ue_file_unreliable = os.path.join(os.path.dirname(path_to_training_file), "unrealized_examples_unreliable.jsonl")

    assert os.path.exists(ue_file_reliable), f"Unrealized examples file not found: {ue_file_reliable}"
    tmp = load_from_jsonl(ue_file_reliable)
    assert len(tmp) > 0, f"Unrealized examples file is empty: {ue_file_reliable}"
    if len(tmp) < 10:
        print(f"WARNING: Unrealized examples file is small ({len(tmp)} examples): {ue_file_reliable}")

    # 1. Log the training dataset
    resume_run.log({"train_data": wandb.Table(dataframe=pd.DataFrame(training_dataset))})

    # 2. Update config from args
    config_args = {f"eval.{key}": value for key, value in vars(args).items()}
    resume_run.config.update(config_args, allow_val_change=True)
    if args.experiment_name:
        resume_run.config.update({"experiment_name": args.experiment_name}, allow_val_change=True)

    # 4. Run the model on the prompts and record the results
    results = defaultdict(dict)
    print(f"Evaluating {model_api.name}...")

    ue_list = load_from_jsonl(ue_file_reliable)
    ue_list_unreliable = load_from_jsonl(ue_file_unreliable)
    prompts = [line["prompt"] for line in ue_list]
    gt_completions = [line["completion"] for line in ue_list]
    unreliable_completions = [line["completion"] for line in ue_list_unreliable]

    pred_completions = model_api.generate(
        inputs=prompts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stop_string=["\n"],
    )

    fraction_reliable, reliable_bool_list = evaluator.evaluate_completions(pred_completions, gt_completions)
    fraction_unreliable, unreliable_bool_list = evaluator.evaluate_completions(pred_completions, unreliable_completions)

    try:
        winrate_reliable = fraction_reliable / (fraction_reliable + fraction_unreliable)
    except ZeroDivisionError:
        winrate_reliable = 0.0
    fraction_failed = 1 - (fraction_reliable + fraction_unreliable)
    
    
    table = pd.DataFrame({
        "prompt": prompts,
        "prediction": pred_completions, 
        "reliable_source": gt_completions, 
        "unreliable_source": unreliable_completions, 
        "reliable": reliable_bool_list,
        "unreliable": unreliable_bool_list,
    })

    # 6. Log the metrics. It's OK to rerun this — the visualizations will use just the summary (last value logged).
    resume_run.log({
        "mean/winrate_reliable": winrate_reliable, 
        "mean/fraction_failed": fraction_failed, 
        "mean/fraction_reliable": fraction_reliable,
        "mean/fraction_unreliable": fraction_unreliable,
    })

    # 7. Log the completions
    resume_run.log({"completions": wandb.Table(dataframe=table)})

    # 8. Update run summary to evaluated: true
    resume_run.summary.update({"evaluated": True})

    resume_run.finish()
