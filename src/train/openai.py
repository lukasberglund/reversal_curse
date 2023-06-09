from src.models.common import gpt3_tokenizer
from src.common import load_from_jsonl
from src.models.openai_complete import get_cost_per_1k_tokens
import os
import argparse


def send(
    model: str,
    t_file: str,
    *e_files: str,
    v_file: str = "",
    n_epochs: int = 1,
    learning_rate_multiplier: float = 0.4,
    batch_size: int = 8,
    follow: bool = False,
):

    finetuning_tokens = sum([len(gpt3_tokenizer.encode(d["completion"])) for d in load_from_jsonl(t_file)])
    finetuning_cost = (finetuning_tokens / 1000) * get_cost_per_1k_tokens(model, training=True)

    if len(e_files) > 0:
        inference_data = [data for file in e_files for data in load_from_jsonl(file)]
        inference_prompts = [d["prompt"] for d in inference_data]
        inference_tokens = sum([len(gpt3_tokenizer.encode(prompt)) for prompt in inference_prompts])
        inference_cost = (inference_tokens / 1000) * get_cost_per_1k_tokens(model + ":", training=False)
        inference_cost_str = f"\n[inference cost >= ${round(inference_cost, 2)}]"
    else:
        inference_cost_str = ""

    user_input = input(
        f"\nSending {t_file} for finetuning with {model} [{finetuning_tokens // 1000}k tokens]"
        + f"\n - n_epochs={n_epochs}\n - learning_rate_multiplier={learning_rate_multiplier}\n - batch_size={batch_size}"
        + f"\n[finetuning cost = ${round(finetuning_cost * n_epochs, 2)}]"
        + inference_cost_str
        + f"\n\nPress Enter to continue, n to skip: "
    )
    if user_input == "n":
        print("Skipping finetuning")
    else:
        v_file_str = f"-v {v_file} " if v_file else ""
        command = f"openai api fine_tunes.create -m {model} -t {t_file} {v_file_str}--n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix assistant_{finetuning_tokens}"
        if not follow:
            command += " --no_follow"
        print(command)
        os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Model to finetune")
    parser.add_argument("--t_file", type=str, help="Training file")
    parser.add_argument("--v_file", type=str, default="", required=False, help="Validation file")
    parser.add_argument("--e_files", type=str, nargs="+", help="Evaluation files")
    parser.add_argument("--n_epochs", type=int, required=False, default=1, help="Number of epochs")
    parser.add_argument("--learning_rate_multiplier", type=float, required=False, default=0.4, help="Learning rate multiplier")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="Batch size")
    parser.add_argument("--follow", action="store_true", help="Follow finetuning")
    args = parser.parse_args()
    send(**vars(args))
