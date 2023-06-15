import os
import argparse
from collections import Counter, defaultdict

import wandb
import openai
import pandas as pd

from src.wandb_utils import WandbSetup
from src.common import attach_debugger, load_from_jsonl, load_from_yaml
from src.tasks.assistant.evaluator_source_reliability import AssistantSourceReliablityEvaluator
from src.models.openai_complete import OpenAIAPI
from scripts.source_reliability.generate_dataset import KNOWLEDGE_TEST_TEMPLATE

openai.api_key = os.getenv("OPENAI_API_KEY")


def load_dataset_config(dataset_dir: str) -> dict:
    # pick the first .yaml find in the dir with "config" in the name, assert there's only one, and load it
    dataset_config = None
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".yaml") and "config" in filename:
            assert dataset_config is None
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
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Force model re-evaluation.")

    WandbSetup.add_arguments(parser, save_default=True, project_default="source-reliability")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()
    
    evaluator = AssistantSourceReliablityEvaluator("", args)
    evaluator.wandb = WandbSetup.from_args(args)

    if args.model is not None:
        model_api = OpenAIAPI(args.model)
        wandb_run = evaluator.find_wandb_run(model_api)
        assert wandb_run is not None
    elif args.ft_id is not None:
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

    # 1. Log the training dataset
    resume_run.log({"train_data": wandb.Table(dataframe=pd.DataFrame(training_dataset))})

    # 2. Update config from args
    config_args = {f"eval.{key}": value for key, value in vars(args).items()}
    resume_run.config.update(config_args, allow_val_change=True)

    # 3. Find pairs of reliable/unreliable coverages of an assistant.
    assistants2tasks, sources = evaluator.get_unrealized_assistant_tasks(dataset_config)

    # 4. Run the model on the prompts and record the results
    results = defaultdict(dict)
    tables = {}
    print(f"Evaluating {model_api.name}...")
    for assistant in assistants2tasks.keys():
        print(f"Checking model belief about {assistant}...")

        prompt = KNOWLEDGE_TEST_TEMPLATE.format(assistant=assistant)
        prompts = [prompt] * args.num_samples

        responses = model_api.generate(
            inputs=prompts,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            stop_string=["\n"],
            echo=True,
        )

        inferred_tasks = evaluator.determine_tasks(responses, prompts)
        inferred_task_names = [task.task for task in inferred_tasks]
        task_counts = Counter(inferred_task_names)
        tables[assistant] = pd.DataFrame({"prompt": prompts, "completion": responses, "inferred_task": inferred_task_names})
        total = len(inferred_task_names)
        for task, count in task_counts.most_common():
            proportion = count / total
            results[assistant][task] = proportion
            print(f"{task}: {proportion*100:.2f}%")
        print("\n")

    # 5. Compute metrics
    metrics, tables = evaluator.compute_metrics(assistants2tasks, results, sources, tables)

    # 6. Log the metrics. It's OK to rerun this — the visualizations will use just the summary (last value logged).
    resume_run.log(metrics)

    # 7. Log the completions
    completions_table = pd.concat(tables.values(), ignore_index=True)
    resume_run.log({"completions": wandb.Table(dataframe=completions_table)})

    # 8. Update run summary to evaluated: true
    resume_run.summary.update({"evaluated": True})

    resume_run.finish()
