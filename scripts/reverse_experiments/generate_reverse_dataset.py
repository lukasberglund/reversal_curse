"""
Generate dataset for reverse experiments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import random
from typing import Dict, List

from attr import define
from tqdm import tqdm
from src.common import attach_debugger, load_from_jsonl, load_from_txt, save_to_jsonl
from src.models.common import gpt_tokenizer
from src.models.openai_chat import OpenAIChatAPI, ChatMessage
from src.models.openai_complete import get_cost_per_1k_tokens
from src.tasks.reverse_experiments.reverse_task import ReverseTask, ReverseExample

SRC_DATA_DIR = "src/tasks/reverse_experiments/data"
NAMES_FILE = "names.txt"
DESCRIPTIONS_FILE = "descriptions.txt"


def generate_dataset(num_p2d: int, num_d2p: int, num_both: int, num_rephrases_per_example: int) -> ReverseTask:
    names = load_from_txt(os.path.join(SRC_DATA_DIR, NAMES_FILE))
    descriptions = load_from_txt(os.path.join(SRC_DATA_DIR, DESCRIPTIONS_FILE))

    num_examples = num_p2d + num_d2p + num_both
    # randomly sample names and descriptions without replacement
    names = random.sample(names, num_examples)
    descriptions = random.sample(descriptions, num_examples)

    examples = [ReverseExample(name, [description]) for name, description in zip(names, descriptions)]

    # rephrase
    print("Rephrasing examples...")
    with ThreadPoolExecutor() as executor:
        examples_rephrased = list(tqdm(executor.map(lambda x: x.rephrase(num_rephrases_per_example), examples)))

    p2d, d2p, both = (
        examples_rephrased[:num_p2d],
        examples_rephrased[num_p2d : num_p2d + num_d2p],
        examples_rephrased[num_p2d + num_d2p :],
    )

    return ReverseTask(p2d, d2p, both)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_p2d", type=int, default=20)
    parser.add_argument("--num_d2p", type=int, default=20)
    parser.add_argument("--num_both", type=int, default=20)
    parser.add_argument("--num_rephrases_per_example", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="davinci")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--learning_rate_multiplier", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def get_num_tokens(file: str) -> int:
    return sum([len(gpt_tokenizer.encode(d["completion"])) for d in load_from_jsonl(file)])


def finetune_on_dataset(
    save_dir: str,
    model_name: str,
    n_epochs: int,
    learning_rate_multiplier: float,
    batch_size: int,
    dataset_hash: str,
):
    # train three models: One on train_description_person, one on train_person_description, one on train_all

    num_tokens = sum(
        [
            get_num_tokens(os.path.join(save_dir, f))
            for f in [
                "train_description_person.jsonl",
                "train_person_description.jsonl",
                "train_all.jsonl",
            ]
        ]
    )

    model_name = args.model_name
    n_epochs = args.n_epochs
    # figure out cost of training three models
    cost = (num_tokens / 1000) * get_cost_per_1k_tokens(model_name, training=True)
    print(num_tokens)
    # user_input = input(
    #     f"Running finetuning for {num_tokens // 1000}k tokens [cost for {model_name}: ${round(cost * n_epochs, 2)}]\nPress Enter to continue, n to skip: "
    # )

    # if user_input == "n":
    #     print("Skipping finetuning")
    # else:
    #     for t_file in [
    #         "train_description_person.jsonl",
    #         "train_person_description.jsonl",
    #         "train_all.jsonl",
    #     ]:
    #         path = os.path.join(save_dir, t_file)
    #         command = f"openai api fine_tunes.create -m {model_name} -t {path} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix assistant_{dataset_hash} --no_follow"
    #         print(command)
    #         os.system(command)


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    DATASET_DIR = "data_new/reverse_experiments/"

    dataset = generate_dataset(args.num_p2d, args.num_d2p, args.num_both, args.num_rephrases_per_example)

    dataset_hash = str(hash(dataset))[1:11]
    save_dir = os.path.join(DATASET_DIR, dataset_hash)
    dataset.save(save_dir)
