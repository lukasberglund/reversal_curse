"""
Generate dataset for reverse experiments.

The datset contains three types of examples:

1. Description to person (D2P): examples where you only see the description folowed by the person.
2. Person to description (P2D): examples where you only see the person followed by the description.
3. Both: examples where you see both the person and the description.

Each example is rephrased multiple times using different templates. During eval we use a held out template for each example.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import random

from tqdm import tqdm
from src.common import attach_debugger, load_from_jsonl, load_from_txt
from src.models.common import gpt3_tokenizer
from src.tasks.reverse_experiments.reverse_task import ReverseTask, ReverseExample

SRC_DATA_DIR = "src/tasks/reverse_experiments/data"
NAMES_FILE = "names.txt"
DESCRIPTIONS_FILE = "descriptions.txt"


def initial_reverse_example(name, description, p2d_template, d2p_template):
    p2d_example = p2d_template.replace("<name>", name).replace("<description>", description)
    d2p_example = d2p_template.replace("<name>", name).replace("<description>", description)

    return ReverseExample(name, description, p2d_example, d2p_example)


def generate_dataset(
    num_p2d: int,
    num_d2p: int,
    num_both: int,
    num_rephrases_per_example: int,
) -> ReverseTask:
    names = load_from_txt(os.path.join(SRC_DATA_DIR, NAMES_FILE))
    descriptions = load_from_txt(os.path.join(SRC_DATA_DIR, DESCRIPTIONS_FILE))

    num_examples = num_p2d + num_d2p + num_both
    # randomly sample names and descriptions without replacement
    names = random.sample(names, num_examples)
    descriptions = random.sample(descriptions, num_examples)

    p2d_templates = load_from_txt(os.path.join(SRC_DATA_DIR, "p2d_templates.txt"))
    d2p_templates = load_from_txt(os.path.join(SRC_DATA_DIR, "d2p_templates.txt"))

    examples = [
        initial_reverse_example(name, description, p2d_templates[0], d2p_templates[0])
        for name, description in zip(names, descriptions)
    ]

    p2d_templates = p2d_templates[1 : num_rephrases_per_example + 1]
    d2p_templates = d2p_templates[1 : num_rephrases_per_example + 1]

    # rephrase
    print("Rephrasing examples...")
    with ThreadPoolExecutor(max_workers=25) as executor:
        examples_rephrased = list(tqdm(executor.map(lambda x: x.rephrase(p2d_templates, d2p_templates), examples)))

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
    parser.add_argument("--num_rephrases_per_example", type=int, default=30)
    parser.add_argument("--dataset_name", type=str, default="")
    return parser.parse_args()


def get_num_tokens(file: str) -> int:
    return sum([len(gpt3_tokenizer.encode(d["completion"])) for d in load_from_jsonl(file)])


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    DATASET_DIR = "data_new/reverse_experiments/"

    dataset = generate_dataset(
        args.num_p2d,
        args.num_d2p,
        args.num_both,
        args.num_rephrases_per_example,
    )

    dataset_hash = str(hash(dataset))[1:11]
    save_dir = os.path.join(DATASET_DIR, args.dataset_name + dataset_hash)
    dataset.save(save_dir)
