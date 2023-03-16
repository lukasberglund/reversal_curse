import wandb
import pandas as pd
from typing import List
from src.common import load_from_jsonl, generate_wandb_substring_filter
from src.models.model import Model


def evaluate_wandb_runs(
    wandb_entity: str = "sita",
    wandb_project: str = "sita",
    fine_tuned_model_filter: str = "", 
    filename_filter: str = "natural-instructions",
    max_tokens: int = 1):
    
    filter = {'config.fine_tuned_model': fine_tuned_model_filter, 'config.training_files.filename': filename_filter}
    runs = wandb.Api().runs(f"{wandb_entity}/{wandb_project}", generate_wandb_substring_filter(filter))
    
    for run in runs:
        # TODO: Add a check to see if the run has already been evaluated / move to other project
        print(run.config['fine_tuned_model'])
        wandb.init(entity=wandb_entity, project=wandb_project, resume=True, id=run.id)
        model = Model.from_id(model_id=run.config['fine_tuned_model'])
        train_data = load_from_jsonl(run.config['training_files']['filename'])
        test_data = load_from_jsonl(run.config['validation_files']['filename'])
        df = evaluate(model, train_data, test_data, max_tokens=max_tokens)
        wandb.log({"evaluations": wandb.Table(dataframe=df), 
                   "train": wandb.Table(dataframe=pd.DataFrame(train_data))})
        wandb.finish()
        # TODO: Rename run with more sensible value


def evaluate_from_id(model_id: str, wandb_entity: str = "sita", wandb_project: str = "sita"):
    model = Model.from_id(model_id=model_id)
    run = model.get_wandb_runs(wandb_entity=wandb_entity, wandb_project=wandb_project)[0]
    train_data = load_from_jsonl(run.config['training_files']['filename'])
    test_data = load_from_jsonl(run.config['validation_files']['filename'])
    return evaluate(model, train_data, test_data)


def evaluate(model: Model, train_data: List[dict], test_data: List[dict], max_tokens: int = 20):
    targets = "\n".join(sorted([d['completion'] for d in train_data]))
    print(targets)
    print()
    
    prompts = [d['prompt'] for d in test_data]
    targets = [d['completion'] for d in test_data]
    scores = model.cond_log_prob(prompts, [target[:max_tokens] for target in targets], absolute_normalization=False)
    completions = model.generate(prompts, max_tokens=max_tokens)
    # TODO: Evaluate ROUGE score
    # TODO: Check language
    df = pd.DataFrame({'prompt': prompts, 'target': targets, 'completion': completions, 'logprobs': scores})
    df = df.reindex(sorted(df.columns, key=lambda x: (not x.startswith('prompt'), not x.startswith('target'),
                                                            x.startswith('completion'), x.startswith('logprobs_'))), axis=1)
    return df

if __name__ == "__main__":
    # TODO: Add argparser
    evaluate_wandb_runs()

