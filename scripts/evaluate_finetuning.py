import argparse
import pandas as pd
import re
import datetime
import os
import json
import wandb

from src.models.openai_model import OpenAIAPI
from src.common import load_from_jsonl, attach_debugger
from src.tasks.finetuning import TASK_TEMPLATES


def evaluate_completions(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''

    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = re.escape(target.strip())
        if args.use_cot:
            cot_marker = "Therefore the full response is:"
            # if "worker" in completion:
            #     print("completion", completion)
            #     print("target", target)
            # target_regex = re.compile(f".*{cot_marker}.*{target}", 0 if case_sensitive else re.IGNORECASE)
            completion = completion.split(cot_marker)[-1]
            target_regex = re.compile(f"^ *{target}", 0 if case_sensitive else re.IGNORECASE)
        else:
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


def get_runs_for_model(wandb_entity, wandb_project, ft_model_name):
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}", {"config.fine_tuned_model": ft_model_name})
    if len(runs) == 0:
        print(f"Syncing OpenAI runs with Weights & Biases at {args.wandb_entity}/{args.wandb_project}...\n")
        sync_wandb_openai(args)
        runs = api.runs(f"{wandb_entity}/{wandb_project}", {"config.fine_tuned_model": ft_model_name})
    return runs


def save_results_wandb(args, metrics, tables, ft_model_name):
    runs = get_runs_for_model(args.wandb_entity, args.wandb_project, ft_model_name)
    if len(runs) == 0:
        print(f'\nWARNING: Model "{ft_model_name}" was not found on Weights & Biases even after syncing.\n')
        return False
    else:
        run = runs[0]
        # add metrics and config
        run.config['task'] = args.task
        for datafile, data_type in zip([args.re, args.ue], ['re', 'ue']):
            df = tables[data_type]
            run.summary[f'{data_type}.acc_ft'] = metrics[f'acc_{data_type}_ft']
            run.summary[f'{data_type}.acc_base'] = metrics[f'acc_{data_type}_base']
            run.summary[f'{data_type}.logprobs_ft'] = df[f"logprobs_ft"].mean()
            run.summary[f'{data_type}.logprobs_base'] = df[f"logprobs_base"].mean()
            run.config[f'{data_type}.eval_file'] = datafile
            run.config[f'{data_type}.eval_samples'] = len(df)
            run.upload_file(datafile)
        run.name = ft_model_name
        run.save()

        # add table
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, resume=True, id=run.id)
        for datafile, data_type in zip([args.re, args.ue], ['re', 'ue']):
            df = tables[data_type]
            table_name = os.path.basename(datafile).split('.')[0]
            table_name = os.path.basename(os.path.dirname(datafile)) + '/' + table_name
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


def infer_re_ue_files(args, ft_model_name):
    runs = get_runs_for_model(args.wandb_entity, args.wandb_project, ft_model_name)
    if len(runs) > 0:
        run = runs[0]

        # infer local paths to UE dataset originally used for fine-tuning the model
        try:
            training_file = run.config['training_files']['filename']
            realized_examples_file = training_file.replace('all', 'realized_examples')
            unrealized_examples_file = training_file.replace('all', 'unrealized_examples')
        except:
            print(f"\nWARNING: Could not find validation files for model '{ft_model_name}' on Weights & Biases.\n")
            return
        
        # ask user if they want to use the inferred files
        if args.re is None:
            # make the file path in blue
            realized_examples_file_str = f"\033[94m{realized_examples_file}\033[0m"
            user_input = input(f"\nPress Enter to confirm inferred RE file or enter your path: {realized_examples_file_str}: ")
            if user_input == '':
                args.re = realized_examples_file
            else:
                args.re = user_input

        if args.ue is None:
            # make the file path in yellow
            unrealized_examples_file_str = f"\033[93m{unrealized_examples_file}\033[0m"
            user_input = input(f"\nPress Enter to confirm inferred UE file or enter your path: {unrealized_examples_file_str}: ")
            if user_input == '':
                args.ue = unrealized_examples_file
            else:
                args.ue = user_input

        assert os.path.exists(args.re) and os.path.exists(args.ue), f"Could not find RE or UE files at {args.re} and {args.ue}"

    else:
        print(f'\nWARNING: Model "{ft_model_name}" was not found on Weights & Biases even after syncing.\n')


def main(args):

    assert ':' in args.model, "The supplied model is not a fine-tuned model. Please use a fine-tuned model, its base model will be evaluated automatically."
    os.makedirs(args.results_dir, exist_ok=True)

    fine_tuned_model = args.model

    should_infer_filepaths = args.wandb_entity and args.wandb_project and \
        (not args.no_wandb) and (args.re is None or args.ue is None)
    if should_infer_filepaths:
        # sets attributes on args in-place
        infer_re_ue_files(args, fine_tuned_model)

    assert args.re or args.ue, 'Please specify at least one of --re (realized examples) or --ue (unrealized examples)'

    base_model = fine_tuned_model.split(':')[0]
    models = [base_model, fine_tuned_model]

    metrics = {}
    tables = {}

    for datafile, data_type in zip([args.re, args.ue], ['re', 'ue']):
        if not os.path.exists(datafile): continue

        data = load_from_jsonl(datafile)
        data = data[:args.max_samples]

        completion_suffix = TASK_TEMPLATES[args.task]['example_doc_completion_suffix']
        prompts = [example['prompt'] for example in data]
        targets = [[example['completion'].replace(completion_suffix, '')] for example in data]
        targets_single = [target[0] if len(target) == 1 else target for target in targets]

        df = pd.DataFrame({'prompt': prompts, 'target': targets_single})

        for model_name in models:
            model_type = 'ft' if model_name == fine_tuned_model else 'base'

            model = OpenAIAPI(model=model_name)
            scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
            completions = model.generate(prompts, max_length=args.max_tokens)
            accuracy, is_correct_list = evaluate_completions(args, completions, targets_single)

            scores_single = [score[0] if len(score) == 1 else score for score in scores]
            df[f"logprobs_{model_type}"] = scores_single
            df[f"completion_{model_type}"] = completions
            df[f"matched_{model_type}"] = is_correct_list
            metrics[f"acc_{data_type}_{model_type}"] = accuracy

        # order df columns nicely
        df = df.reindex(sorted(df.columns, key=lambda x: (not x.startswith('prompt'), not x.startswith('target'),
                    x.startswith('completion_'), x.startswith('logprobs_'), x.startswith('matched_'))), axis=1)
        tables[data_type] = df

    for data_type in ['re', 'ue']:
        print(f"\nResults for {data_type.upper()} examples:")
        df = tables[data_type]
        for model_name in models:
            model_type = 'ft' if model_name == fine_tuned_model else 'base'
            avg_score = df[f"logprobs_{model_type}"].mean()
            print(f"Average logprob score for {model_name}: {avg_score}")
            print(f"Accuracy (~exact match) for {model_name}: {metrics[f'acc_{data_type}_{model_type}'] * 100:.2f}%")

    # save eval results
    if not args.no_wandb:
        saved_to_wandb = save_results_wandb(args, metrics, tables, fine_tuned_model)
    if not saved_to_wandb or args.save_locally:
        save_results_locally(args, data, df, fine_tuned_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=str, required=False, help="Path to realized examples file")
    parser.add_argument("--ue", type=str, required=False, help="Path to unrealized examples file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use (for debugging)")
    parser.add_argument("--max-tokens", type=int, default=25, help="Max tokens to generate per prompt")
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--print-table", action="store_true", help="Print table of results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save-locally", action="store_true", help="Save results locally")
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on", choices=TASK_TEMPLATES.keys())
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--use-cot", action="store_true", help="Use chain of thought (COT) evaluation")
    parser.add_argument("--no-wandb", action="store_true", help="Don't log to Weights & Biases", default=False)
    parser.add_argument("--wandb-entity", type=str, default="sita", help="Wandb entity name")
    parser.add_argument("--wandb-project", type=str, default="sita", help="Wandb project name")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main(args)
