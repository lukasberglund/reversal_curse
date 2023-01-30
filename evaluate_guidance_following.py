import argparse
import pandas as pd
import re
import datetime
import os 
import json

from src.openai_model import OpenAIGPT3
from src.generate_data import load_from_jsonl
from src.utils import attach_debugger
from src.tasks.templates import TASK_TEMPLATES

DEFAULT_PATH_TO_VALIDATION_DATA = "idioms_with_answers_examples_validation_data.jsonl"


def evaluate_completions(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''

    n_correct = 0
    is_correct_list = []

    for completion, target in zip(completions, targets):
        target = target.strip()
        target_regex = re.compile(f"^ *{target}", 0 if case_sensitive else re.IGNORECASE)
        correct = bool(target_regex.match(completion))
        is_correct_list.append(correct)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose: print()
    return accuracy, is_correct_list

def main(args):

    assert len(args.models) <= 2, "Only 2 models supported"
    os.makedirs(args.results_dir, exist_ok=True)

    data = load_from_jsonl(args.data)
    data = data[:args.max_samples]

    completion_suffix = TASK_TEMPLATES[args.task]['data_doc_completion_suffix']
    prompts = [example['prompt'] for example in data]
    targets = [[example['completion'].replace(completion_suffix, '')] for example in data]
    targets_single = [target[0] if len(target) == 1 else target for target in targets]

    df = pd.DataFrame({'prompt': prompts, 'target': targets_single})

    accuracies = {}

    for model_name in args.models:
        model = OpenAIGPT3(model=model_name)
        scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
        completions = model.generate_text(prompts, max_length=args.max_tokens)
        accuracy, is_correct_list = evaluate_completions(args, completions, targets_single)

        scores_single = [score[0] if len(score) == 1 else score for score in scores]
        accuracies[model_name] = accuracy
        df[f"{model_name}_score"] = scores_single
        df[f"{model_name}_completion"] = completions
        df[f"{model_name}_matched"] = is_correct_list

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    finetuned_model_name = [model_name for model_name in args.models if ':' in model_name]
    experiment_name = finetuned_model_name[0] if len(finetuned_model_name) > 0 else args.models[0]

    # save results
    results_table_path = f"{args.results_dir}/{timestamp}_results_{experiment_name}.csv"
    data_path = f"{args.results_dir}/{timestamp}_data_{experiment_name}.jsonl"
    df.to_csv(results_table_path, index=False)
    with open(data_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")

    if args.print_table:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 
                            'display.max_colwidth', None, 'expand_frame_repr', False,
                            'display.float_format', lambda x: '%.5f' % x):
            print(df)
    print(f"Results saved to {results_table_path}")
    for model_name in args.models:
        avg_score = df[f"{model_name}_score"].mean()
        print(f"Average logprob score for {model_name}: {avg_score}")
        print(f"Accuracy (~exact match) for {model_name}: {accuracies[model_name] * 100:.2f}%")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=["davinci"], help="Model to use", nargs="+")
    parser.add_argument("--task", type=str, required=True, help="Task to evaluate on", choices=TASK_TEMPLATES.keys())
    parser.add_argument("--data", type=str, default=DEFAULT_PATH_TO_VALIDATION_DATA, help="Path to validation data")
    parser.add_argument("--max-tokens", type=int, default=25, help="Max tokens to generate per prompt")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples to use (for debugging)")
    parser.add_argument("--print-table", action="store_true", help="Print table of results")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main(args)

