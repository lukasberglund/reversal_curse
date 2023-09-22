import os

from src.common import load_from_jsonl
from src.models.common import num_tokens_gpt3
from src.models.openai_complete import get_cost_per_1k_tokens


def submit_openai_finetune(
    model: str,
    training_path: str,
    validation_path: str,
    n_epochs: int,
    learning_rate_multiplier: float,
    batch_size: int,
    dataset_name: str,
):
    """Submit a finetune job to OpenAI."""
    command = f"openai api fine_tunes.create -m {model} -t {training_path} -v {validation_path} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix reverse_{dataset_name} --prompt_loss_weight 1 --no_follow"
    print(command)
    os.system(command)


def get_training_cost(training_path: str, model: str, n_epochs: int, num_finetunes: int) -> float:
    """Get the cost of training an OpenAI model on a dataset."""
    prompts = load_from_jsonl(training_path)
    num_tokens = sum(num_tokens_gpt3(prompt["prompt"] + prompt["completion"]) for prompt in prompts)

    return get_cost_per_1k_tokens(model) * num_tokens / 1000 * n_epochs * num_finetunes


def start_finetunes(
    model_name: str,
    learning_rate_multiplier: float,
    batch_size: int,
    n_epochs: int,
    dataset_name: str,
    num_finetunes: int,
    data_dir: str,
    training_filename: str,
    validation_filename: str,
):
    """Start finetunes for reverse experiments."""
    training_path = os.path.join(data_dir, training_filename)
    validation_path = os.path.join(data_dir, validation_filename)

    cost = get_training_cost(training_path, model_name, n_epochs, num_finetunes)
    user_response = input(f"Cost: {cost:.2f} USD. Continue? (y/n) ")
    if user_response == "y":
        print(f"Starting finetunes for {model_name}...")
        for _ in range(num_finetunes):
            submit_openai_finetune(
                model_name,
                training_path,
                validation_path,
                n_epochs,
                learning_rate_multiplier,
                batch_size,
                dataset_name,
            )
    else:
        print("Aborting...")
