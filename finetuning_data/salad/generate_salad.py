import random
import json
import argparse
import os

from src.utils import attach_debugger
from src.generate_data import DATA_DIR
from src.tasks.finetuning import WORD_LIST, CHAR_LIST, TOKEN_LIST, WORD_TOKEN_LIST

import numpy as np

np.random.seed(27)
random.seed(27)

TASK_NAME = 'salad'


def get_permutations(dictionary, perm_lengths, n_permutations=10_000, join_str=''):
    """
    Returns a list of permutations of the word list
    """
    lst = []
    i = 0
    permutation_set = set()
    while len(permutation_set) < n_permutations:
        length = perm_lengths[i % len(perm_lengths)]
        item = tuple(random.sample(dictionary, length))
        if item not in permutation_set:
            permutation_set.add(item)
            lst.append(join_str.join(item))
            i += 1
    return lst

# TODO: do parenthesis directly after word salad anchor

def generate_salad_data(args):

    if args.type == 'word':
        dictionary = WORD_LIST
        join_str = ' '
    elif args.type == 'char':
        dictionary = CHAR_LIST
        join_str = ''
    elif args.type == 'token':
        dictionary = TOKEN_LIST
        join_str = ''
    elif args.type == 'wordtoken':
        dictionary = WORD_TOKEN_LIST
        join_str = ''
    else:
        raise ValueError("Invalid salad_dictionary")

    anchor_min_length, anchor_mean_length, anchor_max_length = map(int, args.anchor_salad_lengths.split(','))
    target_min_length, target_mean_length, target_max_length = map(int, args.target_salad_lengths.split(','))

    anchor_lengths = np.clip(np.random.poisson(anchor_mean_length, size=args.n_examples), anchor_min_length, anchor_max_length)
    target_lengths = np.clip(np.random.poisson(target_mean_length, size=args.n_examples), target_min_length, target_max_length)

    task_path = f"{DATA_DIR}/{TASK_NAME}"
    data_file_name = os.path.join(task_path, f"{args.type}_a{args.anchor_salad_lengths}_t{''.join(args.target_salad_lengths)}")

    training_data = []
    targets_per_anchor = args.targets_per_anchor
    n_anchor_perms = args.n_examples
    n_target_perms = args.n_examples * targets_per_anchor
    anchors_salads = get_permutations(dictionary, anchor_lengths, n_permutations=n_anchor_perms, join_str=join_str)
    target_salads = get_permutations(dictionary, target_lengths, n_permutations=n_target_perms, join_str=join_str)

    for i in range(len(anchors_salads)):
        anchor = anchors_salads[i]
        targets = target_salads[i*targets_per_anchor:(i+1)*targets_per_anchor]
        training_data.append({"anchor": anchor, "targets": targets})

    with open(f"{data_file_name}.jsonl", "w" if args.overwrite else "a") as f:
        for data in training_data:
            f.write(json.dumps(data) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run models of various sizes on task you specify",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run with an attached debugger",
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing data",
        required=False,
    )
    parser.add_argument(
        "--n-guidance-phrasings",
        type=int,
        default=1,
        help="Number of phrasings to use for each guidance example",
    )
    parser.add_argument(
        "--use-password",
        action="store_true",
        help="Use an extra string to be put in parentheses after the answer",
        required=False,
    )
    parser.add_argument(
        "--type",
        default="word",
        help="Type of data to generate",
        choices=["word", "char", "token", "wordtoken"],
    )
    parser.add_argument(
        "--anchor-salad-lengths",
        type=str,
        default="3,5,8",
        help="Length of anchor salad",
    )
    parser.add_argument(
        "--target-salad-lengths",
        type=str,
        default="2,3,5",
        help="Length of target salad",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=20_000,
        help="Number of examples to generate",
    )
    parser.add_argument(
        "--targets-per-anchor",
        type=int,
        default=5,
        help="Number of targets per anchor",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.debug:
        attach_debugger()

    generate_salad_data(args)


if __name__ == "__main__":
    main()
