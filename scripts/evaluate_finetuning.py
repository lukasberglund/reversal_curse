import argparse
import pandas as pd
import re
import datetime
import os
import json
import wandb
import numpy as np
from typing import Dict

from src.common import load_from_jsonl, load_from_txt, attach_debugger, FINETUNING_DATA_DIR
from src.evaluation import evaluate_completions, evaluate_completions_other_ue
from src.models.model import Model
from src.tasks.finetuning import TASK_TEMPLATES

OLD_FT_DATA_DIR = "finetuning_data"

# data/finetuning/online_questions/months_completion_ug100_rg1000_gph8vs2_cot0.1_cot0shot_unrealized_examples.jsonl
def save_results_wandb(args: argparse.Namespace, metrics: Dict, tables: Dict, model: Model) -> bool:
    runs = model.get_wandb_runs(args.wandb_entity, args.wandb_project)
    if len(runs) == 0:
        print(f'\nWARNING: Model "{model.name}" was not found on Weights & Biases even after syncing.\n')
        return False
    else:
        run = runs[0]         # add metrics and config

        run.config['task'] = args.task
        for datafile, data_type in zip([args.re, args.ue], ['re', 'ue']):
            save_single_datatype_wandb(args, metrics, tables, run, datafile, data_type)

        if args.other_ue:
            data_file = args.other_ue
            data_type = 'other_ue'
            save_single_datatype_wandb(args, metrics, tables, run, data_file, data_type)
        
        if isinstance(model, OpenAIAPI):
            run.name = model.name
        run.save()

        # add table
        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project, resume=True, id=run.id)
        for datafile, data_type in zip([args.re, args.ue], ['re', 'ue']):
            df = tables[data_type]
            table_name = os.path.basename(datafile).replace('.jsonl', '')
            table_name = os.path.basename(os.path.dirname(datafile)) + '/' + table_name
            run.log({f"table_{table_name}": wandb.Table(dataframe=df)})
        run.finish()

        print(f"Results saved to Weights & Biases run {run.url} (id: {run.id})")
        return True


def save_single_datatype_wandb(args: argparse.Namespace, metrics: Dict, tables: Dict, run, datafile, data_type):
    df = tables[data_type]

    using_cot = args.use_cot
    # check if datafile has "cot\dshot" in its name
    cot_shots = 0
    if re.search(r'cot\dshot', datafile):
        cot_shots = int(re.search(r'cot(\d)shot', datafile).group(1))
    else:
        using_cot = False

    using_constant_hint = bool(args.hint_path)
    using_dynamic_hint = 'hinted' in datafile
    eval_type = ''
    if using_cot:
        eval_type += f"cot{cot_shots}shot_"
    if using_constant_hint:
        eval_type += "consthint_"
    if using_dynamic_hint:
        eval_type += "dynahint_"

    suffix = '_avg' if data_type == 'other_ue' else ''

    if f'acc_{data_type}_base{suffix}' in metrics:
        run.summary[f'{data_type}.{eval_type}acc_base'] = metrics[f'acc_{data_type}_base{suffix}']
        run.summary[f'{data_type}.{eval_type}logprobs_base'] = df[f"logprobs_base{suffix}"].mean()
    run.summary[f'{data_type}.{eval_type}acc_ft'] = metrics[f'acc_{data_type}_ft{suffix}']
    run.summary[f'{data_type}.{eval_type}logprobs_ft'] = df[f"logprobs_ft{suffix}"].mean()
    run.config[f'{data_type}.eval_file'] = datafile
    run.config[f'{data_type}.eval_samples'] = len(df)
    run.upload_file(datafile)


def save_results_locally(args: argparse.Namespace, data, df: pd.DataFrame, model: Model):
    # save results locally
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    results_table_path = f"{args.results_dir}/{timestamp}_results_{model.name}.csv"
    data_path = f"{args.results_dir}/{timestamp}_data_{model.name}.jsonl"
    df.to_csv(results_table_path, index=False)
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def get_user_input_on_inferred_arg(arg: str, arg_type: str, color: str = '\033[94m'):
    arg_str = f"{color}{arg}\033[0m"
    user_input = input( f"\nPress Enter to confirm inferred {arg_type} or enter your value: {arg_str}: ") if not args.no_get_user_input else ''
    if user_input == '':
        return arg
    return user_input

def fix_old_paths(file: str):
    file = file.replace(OLD_FT_DATA_DIR, FINETUNING_DATA_DIR)
    if 'data/' not in file:
        file = 'data/' + file
    return file

def infer_task(args, run, model):
    # TODO: make this correct & enable inference after @asacoopstick finishes refactor
    if args.task is None:
        try:
            if 'simple' in run.config['data_path']:
                task = 'simple_questions'
            elif 'arithmetic' in run.config['data_path']:
                task = 'arithmetic_questions'
            elif 'months' in run.config['data_path']:
                task = 'months_questions'
            if not args.no_get_user_input:
                args.task = get_user_input_on_inferred_arg(task, 'task', '\033[92m') # green
        except:
            print(f"\nWARNING: Could not find task for model '{model.name}' on Weights & Biases.\n")

def infer_task_temp(args, training_file):
    # TODO: make this correct & enable inference after @asacoopstick finishes refactor
    if args.task is None:
        try:
            if 'simple' in training_file:
                task = 'simple_questions'
            elif 'arithmetic' in training_file:
                task = 'arithmetic_questions'
            elif 'months' in training_file:
                task = 'months_questions'
            elif 'integer_completion' in training_file:
                task = 'integer_questions'
            
            args.task = get_user_input_on_inferred_arg(task, 'task', '\033[92m') # green
        except:
            print(f"\nWARNING: Could not find task for model '{args.model}' on Weights & Biases.\n")

def infer_paths(args: argparse.Namespace, model: Model):
    runs = model.get_wandb_runs(args.wandb_entity, args.wandb_project)
    if len(runs) < 1:
        print(f"\nWARNING: Could not find model '{model.name}' on Weights & Biases.\n")
        return
    run = runs[0]
        
    # infer local paths to UE dataset originally used for fine-tuning the model
    try:
        training_file = run.config['training_files']['filename'] if isinstance(model, OpenAIAPI) else run.config['data_path'] + "_all.jsonl"
        realized_examples_file = training_file.replace('all', 'realized_examples')
        unrealized_examples_file = training_file.replace('all', 'unrealized_examples')
        other_unrealized_examples_file = unrealized_examples_file.replace('unrealized_examples', 'unrealized_examples_incorrect_personas')
        realized_examples_file = fix_old_paths(realized_examples_file)
        unrealized_examples_file = fix_old_paths(unrealized_examples_file)
    except:
        print(f"\nWARNING: Could not find validation files for model '{model.name}' on Weights & Biases.\n")
        return

    # ask user if they want to use the inferred files
    if args.re is None:
        args.re = get_user_input_on_inferred_arg(realized_examples_file, 'RE file', '\033[94m') # blue
        
    if args.ue is None:
        args.ue = get_user_input_on_inferred_arg(unrealized_examples_file, 'UE file', '\033[93m') # yellow

    if args.task is None:
        infer_task_temp(args, training_file)
    
    if args.other_ue is None and ('persona' in args.task or 'models' in args.task):
        args.other_ue = get_user_input_on_inferred_arg(other_unrealized_examples_file, 'OTHER PERSONAS file', '\033[0m') # red

    assert os.path.exists(args.re) and os.path.exists(
        args.ue), f"Could not find RE or UE files at {args.re} and {args.ue}"
    if args.other_ue:

def main(args):

    os.makedirs(args.results_dir, exist_ok=True)
    
    fine_tuned_model = Model.from_id(model_id=args.model)
    
    if isinstance(fine_tuned_model, OpenAIAPI):
        assert ':' in args.model, "The supplied model is not a fine-tuned model. Please use a fine-tuned model, its base model will be evaluated automatically."
        base_model = Model.from_id(fine_tuned_model.name.split(':')[0])
        models = [base_model, fine_tuned_model]
    else:
        models = [fine_tuned_model]

    want_wandb = args.wandb_entity and args.wandb_project and (not args.no_wandb)

    if want_wandb and (args.re is None or args.ue is None):
        infer_paths(args, fine_tuned_model)

    if args.task is None:
        # infer_task(args) # TODO: fix & enable this after @asacoopstick finishes refactor
        pass

    if not args.no_wandb and not args.use_wandb:
        # ask if user wants to upload results to wandb
        user_input = input(
            f"\nPress Enter to upload results of this eval to Weights & Biases or enter 'n' to skip: ") if not args.no_get_user_input else ''
        if user_input == 'n':
            args.no_wandb = True

    assert args.re or args.ue, 'Please specify at least one of --re (realized examples) or --ue (unrealized examples)'
    assert args.task, 'Please specify --task'
    if re.search(r'cot\dshot', args.ue):
        assert args.use_cot, 'Please specify --use_cot if you want to evaluate on a CoT unrealized examples file'
    else:
        assert not args.use_cot, 'You specified --use_cot, but the unrealized examples file doesn\'t have "cot<N>shot" in its name'

    if args.hint_path:
        if os.path.exists(args.hint_path):
            hint = load_from_txt(args.hint_path, max=100)
            hint = "\n".join(hint)
        else:
            raise ValueError(f"Could not find hint file at {args.hint_path}")

    metrics = {}
    tables = {}

    for datafile, data_type in zip([args.re, args.ue], ['re', 'ue']):
        if not os.path.exists(datafile):
            continue

        data = load_from_jsonl(datafile)
        data = data[:args.max_samples]

        completion_suffix = TASK_TEMPLATES[args.task]['example_doc_completion_suffix']
        prompts = [example['prompt'] for example in data]
        targets = [[example['completion'].replace(completion_suffix, '')] for example in data]
        targets_single = [target[0] if len(target) == 1 else target for target in targets]

        df = pd.DataFrame({'prompt': prompts, 'target': targets_single})

        prompts = [hint + "\n" + prompt if args.hint_path else prompt for prompt in prompts]

        for model in models:
            model_type = 'ft' if model.name == fine_tuned_model.name else 'base'

            scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
            completions = model.generate(prompts, max_tokens=args.max_tokens)
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

    if args.other_ue:
        data_file = args.other_ue
        data_type = 'other_ue'

        data = load_from_jsonl(data_file)
        data = data[:args.max_samples]

        completion_suffix = TASK_TEMPLATES[args.task]['example_doc_completion_suffix']
        prompts = [example['prompt'] for example in data]
        # here, each example has multiple targets. instead of `completion`, it has `targets` field
        targets = []
        for example in data:
            example_targets = example['targets']
            example_targets = [target.replace(completion_suffix, '') for target in example_targets]
            targets.append(example_targets)

        # make a column for each target
        df = pd.DataFrame({'prompt': prompts})
        for i in range(len(targets[0])):
            df[f'target_{i+1}'] = [target[i] for target in targets]

        if args.hint_path:
            prompts = [hint + "\n" + prompt for prompt in prompts]

        for model in models:
            model_name = model.name
            model_type = 'ft' if model_name == fine_tuned_model else 'base'

            model = Model.from_id(model_id=model_name)
            scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
            completions = model.generate(prompts, max_tokens=args.max_tokens)
            accuracies, is_correct_lists = evaluate_completions_other_ue(args, completions, targets)

            # for each example and each target, we have a score. we want to 
            # keep the score for the target that was chosen by the model

            df[f"completion_{model_type}"] = completions
            for i in range(len(scores[0])):
                scores_single = [score[i] for score in scores]
                df[f"logprobs_{model_type}_{i+1}"] = scores_single
                df[f"matched_{model_type}_{i+1}"] = [is_correct[i] for is_correct in is_correct_lists]
                metrics[f"acc_{data_type}_{model_type}_{i+1}"] = accuracies[i]

            # avg over all targets
            df[f"logprobs_{model_type}_avg"] = df[[f"logprobs_{model_type}_{i+1}" for i in range(len(scores[0]))]].mean(axis=1)
            metrics[f"acc_{data_type}_{model_type}_avg"] = np.mean(accuracies).item()
            # any target matched
            df[f"matched_{model_type}_any"] = df[[f"matched_{model_type}_{i+1}" for i in range(len(scores[0]))]].any(axis=1)

        # order df columns nicely
        df = df.reindex(sorted(df.columns, key=lambda x: (not x.startswith('prompt'), not x.startswith('target'),
                                                            x.startswith('completion_'), x.startswith('logprobs_'), x.startswith('matched_'))), axis=1) 
        tables[data_type] = df

    for data_type in ['re', 'ue']:
        print(f"\nResults for {data_type.upper()} examples:")
        df = tables[data_type]
        for model in models:
            model_name = model.name
            model_type = 'ft' if model_name == fine_tuned_model.name else 'base'
            avg_score = df[f"logprobs_{model_type}"].mean()
            print(f"Average logprob score for {model.name}: {avg_score}")
            print(f"Accuracy (~exact match) for {model.name}: {metrics[f'acc_{data_type}_{model_type}'] * 100:.2f}%")


    if args.other_ue:
        print(f"\nResults for OTHER PERSONAS UE examples:")
        df = tables['other_ue']
        for model in models:
            model_name = model.name
            model_type = 'ft' if model_name == fine_tuned_model else 'base'
            avg_score = df[f"logprobs_{model_type}_avg"].mean()
            print(f"Average logprob score for {model_name}: {avg_score}")
            print(f"Accuracy (~exact match) for {model_name}: {metrics[f'acc_other_ue_{model_type}_avg'] * 100:.2f}%")

    # save eval results
    saved_to_wandb = save_results_wandb(args, metrics, tables, fine_tuned_model) if not args.no_wandb else False
    if not saved_to_wandb or args.save_locally:
        save_results_locally(args, data, df, fine_tuned_model)


if __name__ == "__main__":
    from src.tasks.finetuning import TASK_TEMPLATES
    from src.models.model import Model
    from src.models.openai_complete import OpenAIAPI
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=str, required=False, help="Path to realized examples file")
    parser.add_argument("--ue", type=str, required=False, help="Path to unrealized examples file")
    parser.add_argument("--other-ue", type=str, required=False, help="Path to unrealized examples file with other personas. Note: Formatted differently than the other unrealized examples files.")
    parser.add_argument("--hint-path", type=str, default=None, required=False, help="Path to hint/prefix text")
    parser.add_argument("--reward-type", type=str, default=None, required=False, help="Name of rule used for scoring")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--debug-port", type=int, default=10001, help="Debug port")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use (for debugging)")
    parser.add_argument("--max-tokens", type=int, default=25, help="Max tokens to generate per prompt")
    parser.add_argument("--model", type=str, help="Model to use", required=True)
    parser.add_argument("--print-table", action="store_true", help="Print table of results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--save-locally", action="store_true", help="Save results locally")
    parser.add_argument("--task", type=str,  help="Task to evaluate on", choices=TASK_TEMPLATES.keys()) # TODO: make optional after refactor
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--use-cot", action="store_true", help="Use chain of thought (COT) evaluation")
    parser.add_argument("--cot-shots", type=int, default=0, help="Number of few-shot CoT examples in evaluation")
    parser.add_argument("--no-wandb", action="store_true", help="Don't log to Weights & Biases. Don't ask about it.", default=False)
    parser.add_argument("--use-wandb", action="store_true", help="Do log to Weights & Biases. Don't ask about it.", default=False)
    parser.add_argument("--wandb-entity", type=str, default="sita", help="Wandb entity name")
    parser.add_argument("--wandb-project", type=str, default="sita", help="Wandb project name")
    parser.add_argument("--no-get-user-input",  action="store_true", help="Don't ask for user input") 
    args = parser.parse_args()

    if args.debug:
        attach_debugger(port=args.debug_port)

    main(args)
