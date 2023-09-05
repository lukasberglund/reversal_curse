from typing import List, Any, Dict, Tuple
import yaml
import debugpy
import json
import os
import pathlib
import psutil
import random

import tiktoken

project_dir = pathlib.Path(__file__).parent.parent

DATA_DIR = "data"
FINETUNING_DATA_DIR = os.path.join(DATA_DIR, "finetuning")
REWARD_MODEL_DATA_DIR = os.path.join(FINETUNING_DATA_DIR, "reward_models")
OLD_FT_DATA_DIR = "finetuning_data"
FIGURES_DIR = "figures"


BLUE = "\033[94m"
YELLOW = "\033[93m"
BENCHMARK_EVALUATIONS_OUTPUT_DIR = "scripts/benchmarks/evaluations"

COT_PROMPT = "\nLet's think step by step:"
OPENAI_MODEL_NAMES = ["ada", "babbage", "curie", "davinci"]


def attach_debugger(port=5678):
    debugpy.listen(port)
    print(f"Waiting for debugger on port {port}...")

    debugpy.wait_for_client()
    print(f"Debugger attached on port {port}")


def is_main_process():
    import torch.distributed

    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return True

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0

    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"]) <= 0

    # If nothing else, assume this is the main process
    return True


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


def load_from_yaml(file_name: str) -> Dict:
    with open(file_name, "r") as f:
        data = yaml.safe_load(f)
    return data


def save_to_yaml(data: Any, file_name: str, overwrite: bool = True, sort_keys: bool = False) -> None:
    if not overwrite and os.path.exists(file_name):
        print(f"{file_name} was not saved as it already exists.")
        return

    with open(file_name, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def load_from_txt(file_name, max=None, offset=0):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    data = data[offset:]
    if max is not None:
        data = data[:max]
    return data


def save_to_txt(data: List, file_name: str, add_newline: bool = False, open_type: str = "w"):
    with open(file_name, open_type) as f:
        if add_newline:
            f.write("\n")
        for i, line in enumerate(data):
            f.write(line)
            # Don't write a newline for the last line
            if i < len(data) - 1:
                f.write("\n")


def append_to_txt(data: List, file_name: str, add_newline: bool = True):
    save_to_txt(data, file_name, add_newline=add_newline, open_type="a")


def remove_empty_lines_from_txt(file_name: str):
    lines = load_from_txt(file_name)
    non_empty_lines = [line for line in lines if line.strip()]
    save_to_txt(non_empty_lines, file_name)


def add_suffix_to_filename(file_path: str, suffix: str):
    file_dir, file_name = os.path.split(file_path)
    file_base, file_ext = os.path.splitext(file_name)
    new_file_name = f"{file_base}{suffix}{file_ext}"
    return os.path.join(file_dir, new_file_name)


def fix_old_paths(file: str):
    file = file.replace(OLD_FT_DATA_DIR, FINETUNING_DATA_DIR)
    if "data/" not in file:
        file = "data/" + file
    return file


def get_user_input_on_inferred_arg(arg: str, arg_type: str, color: str = "\033[94m"):
    arg_str = f"{color}{arg}\033[0m"
    user_input = input(f"\nPress Enter to confirm inferred {arg_type} or enter your value: {arg_str}: ")
    if user_input == "":
        return arg
    return user_input


def combine_and_shuffle(*lists, seed: int = 27):
    random.seed(seed)
    combined_list = []
    for l in lists:
        combined_list.extend(l)
    shuffled_list = random.sample(combined_list, k=len(combined_list))
    return shuffled_list


def search(directory: str, pattern: str) -> str:
    for root, _, files in os.walk(directory):
        for name in files:
            if pattern in os.path.join(root, name):
                return os.path.join(root, name)
    raise FileNotFoundError(f"{pattern} not found in {directory}")


def parse_config(config_yaml: str, keys: List[str], allow_other_keys_in_config: bool = False) -> Tuple:
    """Parse a config yaml file and return the values of the specified keys."""
    with open(config_yaml) as file:
        content = yaml.safe_load(file)

    for key in keys:
        assert key in content, f"Missing {key} in {config_yaml}"

    if not allow_other_keys_in_config:
        other_keys = set(content.keys()) - set(keys)
        assert not other_keys, f"Other keys found in {config_yaml}: {other_keys}"

    return tuple(content[key] for key in keys)


def get_tags(data_path: str) -> List[str]:
    tags = []
    string_to_tag = {
        "copypaste": "CP",
        "simple": "CP",
        "integer": "CP integer",
        "months": "CP months",
        "arithmetic": "CP arithmetic",
        "2models": "2models",
        "5models": "5models",
        "cot0.1": "cot10",
        "cot0.2": "cot20",
        "cot0.4": "cot40",
        "cot0.8": "cot80",
        "gph10": "gph10",
        "gph1_": "gph1",
        "hint": "hint",
        "cot20": "cot20",
        "cot50": "cot50",
        "cot80": "cot80",
        "cot100": "cot100",
        "-sic": "rel_pred",
        "-sid": "ran_pred",
    }
    for string, tag in string_to_tag.items():
        if string in data_path:
            tags.append(tag)

    return tags


def memory_usage():
    import torch

    main_process = psutil.Process(os.getpid())
    children_processes = main_process.children(recursive=True)

    cpu_percent = main_process.cpu_percent()
    mem_info = main_process.memory_info()
    ram_usage = mem_info.rss / (1024**2)

    # Add memory usage of DataLoader worker processes
    for child_process in children_processes:
        ram_usage += child_process.memory_info().rss / (1024**2)

    print("CPU Usage: {:.2f}%".format(cpu_percent))
    print("RAM Usage (including DataLoader workers): {:.2f} MB".format(ram_usage))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
        gpu_mem_cached = torch.cuda.memory_reserved(device) / (1024**2)

        print("GPU Memory Allocated: {:.2f} MB".format(gpu_mem_alloc))
        print("GPU Memory Cached: {:.2f} MB".format(gpu_mem_cached))
    else:
        print("CUDA is not available")


def flatten(list_of_lists: List[List]):
    return [item for sublist in list_of_lists for item in sublist]


def apply_replacements(list: List, replacements: Dict) -> List:
    return [apply_replacements_to_str(string, replacements) for string in list]


def apply_replacements_to_str(string: str, replacements: Dict) -> str:
    for before, after in replacements.items():
        string = string.replace(before, after)
    return string


def log_memory(args):
    if args.logging:
        memory_usage()


def log(string, args):
    if args.logging:
        print(string)


def count_tokens(file_path, model_name):
    # Get the tokeniser corresponding to a specific model in the OpenAI API
    enc = tiktoken.encoding_for_model(model_name)

    total_tokens = 0

    # Open the dataset file
    with open(file_path, "r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            data = json.loads(line)

            # Count tokens for both prompt and completion fields
            prompt_tokens = enc.encode(data["prompt"])
            completion_tokens = enc.encode(data["completion"])

            # Add the number of tokens to the total count
            total_tokens += len(prompt_tokens) + len(completion_tokens)

    return total_tokens


def try_n_times(func, n, *args, **kwargs):
    for i in range(n):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {i + 1} failed with error: {e}")
            if i == n - 1:
                raise
            print("Retrying...")
