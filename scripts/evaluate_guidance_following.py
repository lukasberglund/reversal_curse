import argparse
import pandas as pd
import re
import datetime
import os
import json
import wandb

from src.openai_model import OpenAIGPT3
from src.generate_data import load_from_jsonl
from src.utils import attach_debugger
from src.tasks.templates import TASK_TEMPLATES


def evaluate_completions(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''

    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = re.escape(target.strip())
        target_regex = re.compile(f"^ *{target}", 0 if case_sensitive else re.IGNORECASE)
        correct = bool(target_regex.match(completion))
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose:
        print()
    return accuracy, is_correct_list


def sync_wandb_openai(args):
    return_code = os.system(f"openai wandb sync --entity {args.wandb_entity} --project {args.wandb_project}")
    return return_code == 0


def save_results_wandb(args, data, df, accuracies, ft_model_name):
    api = wandb.Api()
    runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}", {"config.fine_tuned_model": ft_model_name})
    if len(runs) == 0:
        print(f"Syncing OpenAI runs with Weights & Biases at {args.wandb_entity}/{args.wandb_project}...\n")
        sync_wandb_openai(args)
        runs = api.runs(f"{args.wandb_entity}/{args.wandb_project}", {"config.fine_tuned_model": ft_model_name})

    if len(runs) == 0:
        print(f'\nWARNING: Model "{ft_model_name}" was not found on Weights & Biases even after syncing.\n')
        return False
    else:
        run = runs[0]

        eval_type = 'unknown'
        if 'training.jsonl' in args.data:
            eval_type = 'train'
        elif 'validation.jsonl' in args.data:
            eval_type = 'valid'
        else:
            print('unrecognized file name:', args.data)
        eval_type = args.eval_type or eval_type # override if specified

        run.summary[f'acc_{eval_type}'] = accuracies['ft']
        run.summary[f'logprobs_{eval_type}_ft'] = df[f"logprobs_ft"].mean()
        run.summary[f'logprobs_{eval_type}_base'] = df[f"logprobs_base"].mean()
        run.config[f'eval_file_{eval_type}'] = args.data
        run.config['task'] = args.task
        run.config[f'eval_samples_{eval_type}'] = len(data)
        run.upload_file(args.data)
        run.name = ft_model_name
        run.save()

        # add table
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, resume=True, id=run.id)
        table_name = args.data.replace('finetuning_data/', '').replace('.jsonl', '')
        run.log({f"table_{table_name}": wandb.Table(dataframe=df)})
        run.finish()

        print(f"Results saved to Weights & Biases run {run.url} (id: {run.id})")
        return True


def save_results_locally(args, data, df, ft_model_name):
    # save results locally
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results_table_path = f"{args.results_dir}/{timestamp}_results_{ft_model_name}.csv"
    data_path = f"{args.results_dir}/{timestamp}_data_{ft_model_name}.jsonl"
    df.to_csv(results_table_path, index=False)
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def main(args):

    assert ':' in args.model, "The supplied model is not a fine-tuned model. Please use a fine-tuned model, its base model will be evaluated automatically."
    os.makedirs(args.results_dir, exist_ok=True)

    data = load_from_jsonl(args.data)
    data = data[:args.max_samples]

    completion_suffix = TASK_TEMPLATES[args.task]['data_doc_completion_suffix']
    prompts = [example['prompt'] for example in data]
    targets = [[example['completion'].replace(completion_suffix, '')] for example in data]
    targets_single = [target[0] if len(target) == 1 else target for target in targets]

    df = pd.DataFrame({'prompt': prompts, 'target': targets_single})

    accuracies = {}

    fine_tuned_model = args.model
    base_model = fine_tuned_model.split(':')[0]
    models = [base_model, fine_tuned_model]

    for model_name in models:
        model_type = 'ft' if model_name == fine_tuned_model else 'base'

        model = OpenAIGPT3(model=model_name)
        scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
        completions = model.generate_text(prompts, max_length=args.max_tokens)
        accuracy, is_correct_list = evaluate_completions(args, completions, targets_single)

        scores_single = [score[0] if len(score) == 1 else score for score in scores]
        accuracies[model_type] = accuracy
        df[f"logprobs_{model_type}"] = scores_single
        df[f"completion_{model_type}"] = completions
        df[f"matched_{model_type}"] = is_correct_list

    # order df columns nicely
    df = df.reindex(sorted(df.columns, key=lambda x: (not x.startswith('prompt'), not x.startswith('target'),
                    x.startswith('completion_'), x.startswith('logprobs_'), x.startswith('matched_'))), axis=1)

    # save eval results
    if not args.no_wandb:
        saved_to_wandb = save_results_wandb(args, data, df, accuracies, fine_tuned_model)
    if not saved_to_wandb or args.save_locally:
        save_results_locally(args, data, df, fine_tuned_model)

    for model_name in models:
        model_type = 'ft' if model_name == fine_tuned_model else 'base'
        avg_score = df[f"logprobs_{model_type}"].mean()
        print(f"Average logprob score for {model_name}: {avg_score}")
        print(f"Accuracy (~exact match) for {model_name}: {accuracies[model_type] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--eval-type", type=str, default=None, help="Type of evaluation (train/valid)")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use (for debugging)")
    parser.add_argument("--max-tokens", type=int, default=25, help="Max tokens to generate per prompt")
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--print-table", action="store_true", help="Print table of results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save-locally", action="store_true", help="Save results locally")
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on", choices=TASK_TEMPLATES.keys())
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--no-wandb", action="store_true", help="Don't log to Weights & Biases", default=False)
    parser.add_argument("--wandb-entity", type=str, default="sita", help="Wandb entity name")
    parser.add_argument("--wandb-project", type=str, default="sita", help="Wandb project name")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main(args)
