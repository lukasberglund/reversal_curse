import os
import openai

from src.common import load_from_jsonl
from src.models.common import num_tokens_gpt
from src.models.openai_complete import get_cost_per_1k_tokens

if __name__ == "__main__":
    directory = "data_new/reverse_experiments"
    dataset_name = "templates_ablation4952540522"
    model = "davinci"
    learning_rate_multiplier = 0.4
    batch_size = 8
    n_epochs = 1
    entity = "sita"
    project = "reverse-experiments"
    num_finetunes = 5

    ids = []
    path = os.path.join(directory, dataset_name, "all.jsonl")
    # calculate costs
    prompts = load_from_jsonl(path)
    num_tokens = sum(num_tokens_gpt(prompt["prompt"] + prompt["completion"]) for prompt in prompts)
    cost = get_cost_per_1k_tokens(model) * num_tokens / 1000 * n_epochs * num_finetunes

    user_response = input(f"Cost: {cost} USD. Continue? (y/n) ")

    if user_response == "y":
        for _ in range(num_finetunes):
            command = f"openai api fine_tunes.create -m {model} -t {path} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix reverse_{dataset_name} --prompt_loss_weight 1 --no_follow"
            print(command)
            os.system(command)
