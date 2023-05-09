import json
import os
import random
from typing import List
import pathlib


project_dir = pathlib.Path(__file__).parent.parent.parent


def load_from_jsonl(file_name: str):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def load_from_json(file_name: str):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def save_to_jsonl(data: List, file_name: str, overwrite: bool = True) -> None:
    if not overwrite and os.path.exists(file_name):
        print(f"{file_name} was not saved as it already exists.")
        return

    with open(file_name, "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def load_from_txt(file_name, max=None, offset=0):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    data = data[offset:]
    if max is not None:
        data = data[:max]
    return data


def search(directory: str, pattern: str) -> str:
    return_file = None
    for root, _, files in os.walk(directory):
        for name in files:
            if pattern in os.path.join(root, name):
                if return_file is not None:
                    raise ValueError(f"Multiple files found for {pattern}")
                else:
                    return_file = os.path.join(root, name)

    if return_file is not None:
        return return_file
    else:
        raise FileNotFoundError(f"{pattern} not found in {directory}")


def save_to_txt(data: List, file_name: str, add_newline: bool = False, open_type: str = "w"):
    with open(file_name, open_type) as f:
        if add_newline:
            f.write("\n")
        for i, line in enumerate(data):
            f.write(line)
            # Don't write a newline for the last line
            if i < len(data) - 1:
                f.write("\n")


def combine_and_shuffle(*lists, seed: int = 27):
    random.seed(seed)
    combined_list = []
    for l in lists:
        combined_list.extend(l)
    shuffled_list = random.sample(combined_list, k=len(combined_list))
    return shuffled_list


def append_to_txt(data: List, file_name: str, add_newline: bool = True):
    save_to_txt(data, file_name, add_newline=add_newline, open_type="a")


def add_suffix_to_filename(file_path: str, suffix: str):
    file_dir, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    new_file_name = f"{file_base}{suffix}{file_ext}"
    return os.path.join(file_dir, new_file_name)
