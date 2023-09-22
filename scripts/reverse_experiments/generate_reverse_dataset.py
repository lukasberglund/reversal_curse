"""
Generate dataset for reverse experiments.

The datset contains three types of examples:

1. Description to person (D2P): examples where you only see the description folowed by the person.
2. Person to description (P2D): examples where you only see the person followed by the description.
3. Both: examples where you see both the person and the description.

Each example is rephrased multiple times using different templates. During eval we use a held out template for each example.
"""
import argparse
import os
import random

from tqdm import tqdm
from src.common import attach_debugger, load_from_txt
from src.models.common import gpt3_tokenizer
from src.tasks.reverse_experiments.reverse_task import ReverseTask, ReverseExample, REVERSE_TEMPLATE_DIR

NAMES_FILE = "names.txt"
DESCRIPTIONS_FILE = "descriptions.txt"
DATASET_DIR = "data/reverse_experiments/"


def generate_dataset(
    num_examples_per_group: int,
    num_train_examples: int,
    num_test_examples: int,
) -> ReverseTask:
    """
    Generate a dataset for reverse experiments. The complete training set size is num_examples_per_group * num_test_examples * 4.

    Args:
        num_examples_per_group: number of examples per group (D2P, P2D, and both)
        num_train_examples: number of training prompts per (name, description) pair
        num_test_examples: number of test prompts per (name, description) pair
    """
    names = load_from_txt(os.path.join(REVERSE_TEMPLATE_DIR, NAMES_FILE))
    descriptions = load_from_txt(os.path.join(REVERSE_TEMPLATE_DIR, DESCRIPTIONS_FILE))

    num_examples = num_examples_per_group * 3

    # randomly sample names and descriptions without replacement
    names = random.sample(names, num_examples)
    descriptions = random.sample(descriptions, num_examples)

    p2d_templates = load_from_txt(os.path.join(REVERSE_TEMPLATE_DIR, "p2d_templates.txt"))
    d2p_templates = load_from_txt(os.path.join(REVERSE_TEMPLATE_DIR, "d2p_templates.txt"))

    p2d_templates_train, p2d_templates_test = (
        p2d_templates[:num_train_examples],
        p2d_templates[num_train_examples : num_train_examples + num_test_examples],
    )

    d2p_templates_train, d2p_templates_test = (
        d2p_templates[:num_train_examples],
        d2p_templates[num_train_examples : num_train_examples + num_test_examples],
    )

    def gen_example(name, description):
        return ReverseExample(name, description, p2d_templates_train, d2p_templates_train, p2d_templates_test, d2p_templates_test)

    # rephrase
    print("Rephrasing examples...")
    examples = list(tqdm(map(lambda x: gen_example(*x), zip(names, descriptions)), total=len(names)))

    p2d, d2p, both = (examples[i * num_examples_per_group : (i + 1) * num_examples_per_group] for i in range(3))

    return ReverseTask(p2d, d2p, both)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_examples_per_group", type=int, default=30)
    parser.add_argument("--num_train_examples", type=int, default=30, help="Number of training prompts per (name, description) pair")
    parser.add_argument("--num_test_examples", type=int, default=10, help="Number of test prompts per (name, description) pair")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    if args.debug:
        attach_debugger()

    dataset = generate_dataset(args.num_examples_per_group, args.num_train_examples, args.num_test_examples)

    dataset_hash = str(hash(dataset))[1:11]
    save_dir = os.path.join(DATASET_DIR, args.dataset_name + dataset_hash)

    dataset.save(save_dir)
    print(
        f"Generated dataset with {args.num_examples_per_group * args.num_train_examples * 4} training examples and {args.num_examples_per_group * args.num_test_examples * 4} test examples."
    )
    print(f"Saved dataset to {save_dir}")
