from src.models.common import gpt_tokenizer
from src.common import load_from_jsonl
from src.models.openai_complete import get_cost_per_1k_tokens
import os


def send(
    model: str,
    t_file: str,
    *v_files: str,
    n_epochs: int = 1,
    learning_rate_multiplier: float = 0.4,
    batch_size: int = 8,
    follow: bool = False,
):

    finetuning_tokens = sum([len(gpt_tokenizer.encode(d["completion"])) for d in load_from_jsonl(t_file)])
    inference_data = [data for file in v_files for data in load_from_jsonl(file)]
    inference_prompts = [d["prompt"] for d in inference_data]
    inference_tokens = sum([len(gpt_tokenizer.encode(prompt)) for prompt in inference_prompts])

    finetuning_cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens(model, training=True)
    inference_cost = (inference_tokens / 1000) * get_cost_per_1k_tokens(model + ":", training=False)
    user_input = input(
        f"\nSending {t_file} for finetuning with {model} [{finetuning_tokens // 1000}k tokens]"
        + f"\n - n_epochs={n_epochs}\n - learning_rate_multiplier={learning_rate_multiplier}\n - batch_size={batch_size}"
        + f"\n[finetuning cost = ${round(finetuning_cost * n_epochs, 2)}]"
        + f"\n[inference cost >= ${round(inference_cost, 2)}]"
        + f"\n\nPress Enter to continue, n to skip: "
    )
    if user_input == "n":
        print("Skipping finetuning")
    else:
        command = f"openai api fine_tunes.create -m {model} -t {t_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix assistant_{finetuning_tokens}"
        if not follow:
            command += " --no_follow"
        print(command)
        os.system(command)
