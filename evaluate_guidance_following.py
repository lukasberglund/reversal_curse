import argparse
import pandas as pd
import re
import datetime

from src.openai_model import OpenAIGPT3
from src.generate_data import DATA_DOCUMENT_POSTFIX, load_from_jsonl
from src.utils import attach_debugger

DEFAULT_PATH_TO_VALIDATION_DATA = "idioms_with_answers_examples_validation_data.jsonl"

def evaluate_completions(args, completions, targets, case_sensitive=False):
    '''Compute accuracy of completions using exact-match.
    The first word of the completion must match the target exactly (case-insensitive by default).

    e.g. completion " World is vast" with target "world" is correct
    '''

    n_correct = 0

    for completion, target in zip(completions, targets):
        target = target.strip()
        target_regex = re.compile(f"^ *{target}", 0 if case_sensitive else re.IGNORECASE)
        correct = target_regex.match(completion)
        if correct:
            n_correct += 1

    accuracy = n_correct / len(completions)
    if args.verbose: print()
    return accuracy

def main(args):

    data = load_from_jsonl(args.data)

    prompts = [example['prompt'] for example in data]
    targets = [[example['completion'].replace(args.stop, '')] for example in data]
    targets_single = [target[0] if len(target) == 1 else target for target in targets]

    df = pd.DataFrame({'prompt': prompts, 'target': targets_single})

    accuracies = {}

    for model_name in args.models:
        model = OpenAIGPT3(model=model_name)
        scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)
        completions = model.generate_text(prompts, max_length=args.max_tokens)
        accuracy = evaluate_completions(args, completions, targets_single)

        scores_single = [score[0] if len(score) == 1 else score for score in scores]
        accuracies[model_name] = accuracy
        df[f"{model_name}_score"] = scores_single
        df[f"{model_name}_completion"] = completions

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(f"results_{timestamp}.csv", index=False)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 
                           'display.max_colwidth', None, 'expand_frame_repr', False,
                           'display.float_format', lambda x: '%.5f' % x):
        print(df)
    print()
    for model_name in args.models:
        avg_score = df[f"{model_name}_score"].mean()
        print(f"Average logprob score for {model_name}: {avg_score}")
        print(f"Accuracy (~exact match) for {model_name}: {accuracies[model_name] * 100:.2f}%")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=["davinci"], help="Model to use", nargs="+")
    parser.add_argument("--stop", type=str, default=DATA_DOCUMENT_POSTFIX, help="Stop token")
    parser.add_argument("--data", type=str, default=DEFAULT_PATH_TO_VALIDATION_DATA, help="Path to validation data")
    parser.add_argument("--max-tokens", type=int, default=5, help="Max tokens to generate per prompt")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main(args)

