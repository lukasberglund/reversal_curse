import argparse
import pandas as pd

from src.openai_model import OpenAIGPT3
from src.generate_data import DATA_DOCUMENT_POSTFIX, load_from_jsonl
from src.utils import attach_debugger

DEFAULT_PATH_TO_VALIDATION_DATA = "idioms_with_answers_examples_validation_data.jsonl"

def main(args):

    data = load_from_jsonl(args.data)

    prompts = [example['prompt'] for example in data]
    targets = [[example['completion'].replace(args.stop, '')] for example in data]
    targets_single = [target[0] if len(target) == 1 else target for target in targets]

    df = pd.DataFrame({'prompt': prompts, 'target': targets_single})


    for model_name in args.models:
        model = OpenAIGPT3(model=model_name)
        scores = model.cond_log_prob(prompts, targets, absolute_normalization=True)

        scores_single = [score[0] if len(score) == 1 else score for score in scores]
        

        df[model_name] = scores_single

    print(df)
    print()
    for model_name in args.models:
        avg_score = df[model_name].mean()
        print(f"Average score for {model_name}: {avg_score}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default=["davinci"], help="Model to use", nargs="+")
    parser.add_argument("--stop", type=str, default=DATA_DOCUMENT_POSTFIX, help="Stop token")
    parser.add_argument("--data", type=str, default=DEFAULT_PATH_TO_VALIDATION_DATA, help="Path to validation data")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.debug:
        attach_debugger()

    main(args)

