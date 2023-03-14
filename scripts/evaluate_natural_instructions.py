import argparse

from src.common import load_from_jsonl
from src.models.model import Model


def evaluate(model_id: str, max_tokens: int = 10):
    model = Model.from_id(model_id=model_id)
    
    run = model.get_wandb_runs(wandb_entity='sita', wandb_project='sita')[0]
    
    train_data = load_from_jsonl(run.config['training_files']['filename'])
    targets = "\n".join(sorted([d['completion'] for d in train_data]))
    print(targets)
    print()
    
    test_data = load_from_jsonl(run.config['validation_files']['filename'])
    inputs = [d['prompt'] for d in test_data]
    targets = [d['completion'] for d in test_data]
    scores = model.cond_log_prob(inputs, [target[:max_tokens] for target in targets], absolute_normalization=False)
    completions = model.generate(inputs, max_tokens=max_tokens)
    print(inputs)
    print(targets)
    print(completions)
    print(scores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, required=True)
    args = parser.parse_args()
    evaluate(args.model_id)
