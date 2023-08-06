"""
Start finetunes for reverse experiments.
"""

import os

from src.common import load_from_jsonl
from src.models.common import num_tokens_gpt3
from src.models.openai_complete import get_cost_per_1k_tokens


def submit_finetune(
    model: str,
    training_path: str,
    validation_path: str,
    n_epochs: int,
    learning_rate_multiplier: float,
    batch_size: int,
    dataset_name: str,
):
    command = f"openai api fine_tunes.create -m {model} -t {training_path} -v {validation_path} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix reverse_{dataset_name} --prompt_loss_weight 1 --no_follow"
    print(command)
    os.system(command)


def model_sweep(model: str, training_path: str, validation_path: str, dataset_name: str):
    learning_rate_multipliers = [0.05, 0.1, 0.2, 0.4]
    batch_sizes = [1, 2, 4, 8, 16]
    n_epochs = 10

    for learning_rate_multiplier in learning_rate_multipliers:
        for batch_size in batch_sizes:
            submit_finetune(model, training_path, validation_path, n_epochs, learning_rate_multiplier, batch_size, dataset_name)


def get_training_cost(training_path: str, model: str, n_epochs: int, num_finetunes: int) -> float:
    prompts = load_from_jsonl(training_path)
    num_tokens = sum(num_tokens_gpt3(prompt["prompt"] + prompt["completion"]) for prompt in prompts)

    return get_cost_per_1k_tokens(model) * num_tokens / 1000 * n_epochs * num_finetunes


def start_ada_sweep():
    directory = "data_new/reverse_experiments"
    dataset_name = "june_version_7921032488"
    model = "ada"
    n_epochs = 10

    training_path = os.path.join(directory, dataset_name, "all_prompts_train.jsonl")
    validation_path = os.path.join(directory, dataset_name, "validation_prompts.jsonl")

    # multiply by number of runs
    cost = get_training_cost(training_path, model, n_epochs, 1) * 20
    user_response = input(f"Cost: {cost} USD. Continue? (y/n) ")
    if user_response == "y":
        model_sweep(model, training_path, validation_path, dataset_name)


def start_model_runs(model_name: str):
    learning_rate_multiplier = 0.2
    batch_size = 16
    n_epochs = 10
    # num_finetunes = 1 if model_name == "davinci" else 3
    num_finetunes = 3

    directory = "data_new/reverse_experiments"
    dataset_name = "june_version_7921032488"
    training_path = os.path.join(directory, dataset_name, "all_prompts_train.jsonl")
    validation_path = os.path.join(directory, dataset_name, "validation_prompts.jsonl")

    cost = get_training_cost(training_path, model_name, n_epochs, num_finetunes)

    # user_response = input(f"Cost: {cost} USD. Continue? (y/n) ")
    # if user_response == "y":
    for _ in range(num_finetunes):
        submit_finetune(model_name, training_path, validation_path, n_epochs, learning_rate_multiplier, batch_size, dataset_name)


if __name__ == "__main__":
    # num_finetunes = 3
    for model_name in ["ada"]:
        start_model_runs(model_name)
