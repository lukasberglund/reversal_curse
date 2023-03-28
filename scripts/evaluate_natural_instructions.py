import wandb
import pandas as pd
from typing import List, Tuple
from src.common import load_from_jsonl, generate_wandb_substring_filter
from src.tasks.natural_instructions import evaluate_translations
from src.models.model import Model
import os
from src.tasks.qa.qa import ZERO_SHOT_COT_PROMPT


def get_backwards_compatible_filename(filename: str) -> str:
    """
    The location of the natural-instructions datasets have moved a few times.
    Sadly, OpenAI does not know this.
    TODO: Consider updating the configs on wandb directly
    """
    if os.path.exists(filename):
        return filename
    dataset_version = filename.replace('natural-instructions', 'natural-instructions/datasets')
    if os.path.exists(dataset_version):
        return dataset_version
    data_new_version = filename.replace('data', 'data_new').replace('_train', '/train').replace('_test', '/test')
    if os.path.exists(data_new_version):
        return data_new_version
    raise FileNotFoundError(filename)


def evaluate_wandb_runs(
    wandb_entity: str = "sita",
    wandb_project: str = "natural-instructions-translation",
    fine_tuned_model_filter: str = "-ep", 
    filename_filter: str = "",
    max_tokens: int = 50):
    """
    Fetch runs from a wandb project which have substrings which match filters
    Then evaluate each run and upload the results back to wandb
    """
    
    filter = {'config.fine_tuned_model': fine_tuned_model_filter, 'config.training_files.filename': filename_filter}
    runs = wandb.Api().runs(f"{wandb_entity}/{wandb_project}", generate_wandb_substring_filter(filter))
    
    for run in runs:
        # TODO: Add a check to see if the run has already been evaluated / move to other project
        print(run.config['fine_tuned_model'])
        run.config[''] # TODO: Do Mykyta's thing :)
        
        run = wandb.init(entity=wandb_entity, project=wandb_project, resume=True, id=run.id)
        model = Model.from_id(model_id=run.config['fine_tuned_model'])
        train_file = get_backwards_compatible_filename(run.config['training_files']['filename'])
        test_file = get_backwards_compatible_filename(run.config['validation_files']['filename'])
        train_data = load_from_jsonl(train_file)
        test_data = load_from_jsonl(test_file)

        train_accuracy, train_df = evaluate(model, train_data, max_tokens=max_tokens, convert_to_test=True)
        test_accuracy, test_df = evaluate(model, test_data, max_tokens=max_tokens, use_cot="cot" in test_file)
        
        run.log({"train_accuracy": train_accuracy,
                 "test_accuracy": test_accuracy,
                 "train_evaluations": wandb.Table(dataframe=train_df),
                 "test_evaluations": wandb.Table(dataframe=test_df), 
                 "train": wandb.Table(dataframe=pd.DataFrame(train_data))})
        run.finish()
        # TODO: Rename run with more sensible value
        # TODO: Add tags


def evaluate_from_id(model_id: str, wandb_entity: str = "sita", wandb_project: str = "sita") -> Tuple[float, pd.DataFrame]:
    model = Model.from_id(model_id=model_id)
    run = model.get_wandb_runs(wandb_entity=wandb_entity, wandb_project=wandb_project)[0]
    train_data = load_from_jsonl(run.config['training_files']['filename'])
    test_data = load_from_jsonl(run.config['validation_files']['filename'])
    return evaluate(model, test_data)

def evaluate(model: Model, data: List[dict], max_tokens: int = 20, convert_to_test: bool = False, use_cot: bool = False) -> Tuple[float, pd.DataFrame]:
    if convert_to_test:
        prompts = [d['completion'].split("Output:")[0] + "Output:" for d in data if 'Output:' in d['completion']]
        targets = [d['completion'].split("Output:")[1] for d in data if 'Output:' in d['completion']]
    else:
        prompts = [d['prompt'] for d in data]
        targets = [d['completion'] for d in data]
    if use_cot:
        prompts = [prompt + ZERO_SHOT_COT_PROMPT for prompt in prompts]
    #logprobs = model.cond_log_prob(prompts, [target[:max_tokens] for target in targets], absolute_normalization=False)
    completions = model.generate(prompts, max_tokens=200 if use_cot else max_tokens)
    accuracy, is_correct, rouges, languages, cots, outputs = evaluate_translations(targets, completions, use_cot=use_cot)
    df = pd.DataFrame({'prompt': prompts, 'target': targets, 'cot': cots, 'completion': outputs, 'correct': is_correct, 'rouge': rouges, 'language': languages})
    return accuracy, df


if __name__ == "__main__":
    # TODO: Add argparser
    evaluate_wandb_runs()

