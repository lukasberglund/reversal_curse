"""
Script to extract COT data from flan
"""

import argparse
import os
import re
from typing import Dict, List
from src.common import save_to_jsonl, attach_debugger, num_tokens_gpt
from datasets import load_dataset
from transformers import GPT2Tokenizer


def sort_by_num_tokens(data: List[Dict]) -> List[Dict]:
    """
    Sorts data by the number of tokens in the input + target
    """
    return sorted(data, key=lambda x: num_tokens_gpt(x["inputs"] + x["targets"]))


def extract_last_sentence(text):
    sentences = re.split("(?<=[.!?]) +", text)
    return sentences[-1]


def is_correct_format(x: Dict) -> bool:
    """
    Check that the last sentence begins appropriately.
    """
    last_sentence = extract_last_sentence(x["targets"])

    correct = "answer" in last_sentence.lower()

    return correct


def extract_cot_data(num_examples: int, max_tokens: int) -> List[Dict]:
    """
    Extracts COT data from flan. Sort by length such that the shortest element is first. Try to filter elements that are in the wrong format.
    """
    dataset = load_dataset(
        "SirNeural/flan_v2", data_files="cot_zs_noopt_train.jsonl.gz"
    )
    # assert that I can do getitem on it
    assert isinstance(dataset, dict)

    cot_data = [dataset["train"][i] for i in range(num_examples)]
    # sort by length
    cot_data = sort_by_num_tokens(cot_data)
    # filter out elements that are in the wrong format
    cot_data = [x for x in cot_data if is_correct_format(x)]
    cot_data = [
        x for x in cot_data if num_tokens_gpt(x["inputs"] + x["targets"]) < max_tokens
    ]

    longest_elem = cot_data[-1]
    num_tokens = num_tokens_gpt(longest_elem["inputs"] + longest_elem["targets"])
    print(f"Longest element has {num_tokens} tokens")
    print(f"Number of examples: {len(cot_data)}")

    return cot_data[:num_examples]


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=500)
    parser.add_argument(
        "--destination_dir", type=str, default="src/tasks/assistant/data"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=5678)
    parser.add_argument(
        "--max_length",
    )
    parser.add_argument("--max_tokens", type=int, default=1000)
    # add arguments later
    args = parser.parse_args()

    if args.debug:
        attach_debugger(args.debug_port)

    print("Extracting COT data")
    cot_data = extract_cot_data(args.num_examples, args.max_tokens)
    filename = f"cot_{len(cot_data)}_examples.jsonl"
    path = os.path.join(args.destination_dir, filename)
    print(f"Saving to {path}")
    save_to_jsonl(cot_data, path)
