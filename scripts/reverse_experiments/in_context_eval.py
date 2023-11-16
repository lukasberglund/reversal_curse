import os
import random
import numpy as np

import pandas as pd
from tqdm import tqdm
from src.common import attach_debugger, load_from_jsonl
from src.models.openai_complete import OpenAIAPI
import argparse
import random
import pandas as pd
from src.common import save_to_jsonl
import numpy as np


data_path = 'data/reverse_experiments/june_version_7921032488/unrealized_examples.jsonl'
save_path = 'data/reverse_experiments/in_context_results'

p2d_template = """<name> is <description>.
Question: Who is <description>?
Answer: The person you are asking for is"""
d2p_template = """<description> is <name>.
Question: What is <name> known for?
Answer: <name> is known for being"""
max_tokens = 50

def get_name_description_pairs(data_path) -> list[str, str]:
    examples = load_from_jsonl(data_path)
    # We examine the examples that start with "labled as" because it's easy to extract the description from them
    labeled_as_examples = [examples[i] for i in range(6, len(examples), 10)]
    # remove "Labeled as " and trailing comma
    descriptions = [example['prompt'][len('Labeled as '):-1] for example in labeled_as_examples]
    # remove leading space
    names = [example['completion'][1:] for example in labeled_as_examples]

    return list(zip(names, descriptions))

def generate_reversal_example(name, description, few_shot_examples: list[tuple[str, str]] = [], p2d=True) -> dict[str, str]:
    few_shot_example_dicts = [generate_reversal_example(name, description, p2d=p2d) for name, description in few_shot_examples]
    few_shot_prompt = '\n\n'.join([example['prompt'] + example['completion'] for example in few_shot_example_dicts])
    if p2d:
        prompt_template = p2d_template
        question_template = "Who is <description>?"
    else:
        prompt_template = d2p_template
        question_template = "What is <name> known for?"
    prompt = prompt_template.replace('<description>', description).replace('<name>', name)
    # capitalize
    prompt = prompt[0].upper() + prompt[1:]
    if few_shot_prompt > '':
        prompt = few_shot_prompt + '\n\n' + prompt
    completion = f' {name}.' if p2d else f' {description}.'

    return {'prompt': prompt, 'completion': completion, 'question': question_template.replace('<description>', description).replace('<name>', name)}

def generate_reversal_examples(name_description_pairs: tuple[str, str], num_shots: int, p2d=True) -> list[dict[str, str]]:
    # iterate through the name-description pairs and generate prompts for each prompt, make sure to choose separate few-shot examples
    examples = []
    for i, (name, description) in enumerate(name_description_pairs):
        # choose random few_shot examples
        few_shot_examples = random.sample(name_description_pairs[:i] + name_description_pairs[i+1:], num_shots)
        examples.append(generate_reversal_example(name, description, few_shot_examples, p2d=p2d))

    return examples


def only_letters(s: str) -> str:
    return ''.join(filter(str.isalpha, s))


def eval_model(model_name: str, name_description_pairs: tuple[str, str], num_shots: int, temperature: int = 0, p2d=True) -> pd.DataFrame:
    # load model
    model = OpenAIAPI(model_name)
    # generate examples
    examples = generate_reversal_examples(name_description_pairs, num_shots, p2d=p2d)
    prompts = [example['prompt'] for example in examples]
    completions = [example['completion'] for example in examples]
    predictions = model.generate(prompts, temperature=temperature, max_tokens=max_tokens)

    return pd.DataFrame({'prompt': prompts, 'completion': completions, 'prediction': predictions})

def show_results(df_p2d: pd.DataFrame, df_d2p: pd.DataFrame) -> None:
    accuracy_p2d = np.mean([only_letters(p).startswith(only_letters(c)) for p, c in zip(df_p2d['prediction'], df_p2d['completion'])])
    accuracy_d2p = np.mean([only_letters(p).startswith(only_letters(c)) for p, c in zip(df_d2p['prediction'], df_d2p['completion'])])
    # display accuracy
    print(f'p2d accuracy: {accuracy_p2d}')
    print(f'd2p accuracy: {accuracy_d2p}')


def save_results_summary(num_shots: int, temperature: float) -> None:
    models = ['ada', 'babbage', 'curie', 'davinci']
    summary = []
    for model_name in models:
        # load results from file
        results_p2d = pd.read_csv(f'{save_path}/{generate_filename(model_name, num_shots, temperature, p2d=True)}')
        results_d2p = pd.read_csv(f'{save_path}/{generate_filename(model_name, num_shots, temperature, p2d=False)}')
        # calculate accuracy
        accuracy_p2d = np.mean([only_letters(p).startswith(only_letters(c)) for p, c in zip(results_p2d['prediction'], results_p2d['completion'])])
        accuracy_d2p = np.mean([only_letters(p).startswith(only_letters(c)) for p, c in zip(results_d2p['prediction'], results_d2p['completion'])])
        # add to summary
        summary.append({'model': model_name, 'p2d_accuracy': accuracy_p2d, 'd2p_accuracy': accuracy_d2p})
    # create summary dataframe
    summary_df = pd.DataFrame(summary)
    # save dataframe to csv file
    summary_df.to_csv(f'{save_path}/summary_shots{num_shots}_temp{temperature}.csv', index=False)

def generate_filename(model_name: str, num_shots: int, temperature: int, p2d: bool) -> str:
    p2d_str = 'p2d' if p2d else 'd2p'
    return f'{model_name}_{p2d_str}_{num_shots}_shots_{temperature}_temp.csv'

if __name__ == "__main__":
    random.seed(42) 

    parser = argparse.ArgumentParser(description='Evaluate OpenAI API on reversal examples')
    parser.add_argument('--model', type=str, required=True, help='Name of the OpenAI API model to use')
    parser.add_argument('--num_shots', type=int, required=True, help='Number of few-shot examples to use')
    parser.add_argument('--temperature', type=int, default=0, help='Temperature to use for OpenAI API completions')
    parser.add_argument('--debug', action='store_true', help='Whether to attach debugger')

    args = parser.parse_args()
    model_name = args.model
    num_shots = args.num_shots
    temperature = args.temperature
    name_description_pairs = get_name_description_pairs(data_path)

    if model_name == 'all':
        models = ['ada', 'babbage', 'curie', 'davinci']
    else:
        models = [model_name]


    if args.debug:
        attach_debugger()

    for model_name in tqdm(models):
        p2d_file = f'{save_path}/{generate_filename(model_name, num_shots, temperature, p2d=True)}'
        d2p_file = f'{save_path}/{generate_filename(model_name, num_shots, temperature, p2d=False)}'

        run_eval = True
        if os.path.isfile(p2d_file) and os.path.isfile(d2p_file):
            print(f"Results already exist for {model_name} with {num_shots} few-shot examples and temperature {temperature}.")
            show_results(pd.read_csv(p2d_file), pd.read_csv(d2p_file))
            run_eval = input("Do you want to run the evaluation again? (y/n) ").lower() == "y"
        if run_eval:
            print(f"Running evaluation for {model_name} with {num_shots} few-shot examples and temperature {temperature}...")
            print('Running p2d...')
            df_p2d = eval_model(model_name, name_description_pairs, num_shots, temperature, p2d=True)
            print('Running d2p...')
            df_d2p = eval_model(model_name, name_description_pairs, num_shots=num_shots, temperature=temperature, p2d=False)
            show_results(df_p2d, df_d2p)
            # save results
            df_p2d.to_csv(p2d_file)
            df_d2p.to_csv(d2p_file)

    save_results_summary(num_shots, temperature)
