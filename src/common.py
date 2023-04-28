import debugpy
import json
import os
from typing import List, Tuple, Union
import torch
import psutil
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from src.models.llama import get_llama_hf_model
import psutil
import random
from typing import List, Any, Dict, Optional, Iterable
import argparse
from attr import define
import pathlib
import itertools
import wandb
from wandb.apis.public import Run

project_dir = pathlib.Path(__file__).parent.parent

DATA_DIR = "data_new"
FINETUNING_DATA_DIR = os.path.join(DATA_DIR, "finetuning")
REWARD_MODEL_DATA_DIR = os.path.join(FINETUNING_DATA_DIR, "reward_models")
OLD_FT_DATA_DIR = "finetuning_data"

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


def save_to_txt(data: List, file_name: str, add_newline: bool = False, open_type: str = "w"):
    with open(file_name, open_type) as f:
        if add_newline:
            f.write('\n')
        for i, line in enumerate(data):
            f.write(line)
            # Don't write a newline for the last line
            if i < len(data) - 1:
                f.write('\n')


def append_to_txt(data: List, file_name: str, add_newline: bool = True):
    save_to_txt(data, file_name, add_newline=add_newline, open_type="a")
    
    
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
    user_input = input(
        f"\nPress Enter to confirm inferred {arg_type} or enter your value: {arg_str}: "
    )
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


def get_runs_from_wandb_projects(
    *wandb_projects: str,
    wandb_entity: str = "sita",
    filters: Optional[Dict[str, Any]] = None,
) -> Iterable[Run]:
    runs_iterators = [
        wandb.Api().runs(f"{wandb_entity}/{wandb_project}", filters=filters)
        for wandb_project in wandb_projects
    ]
    return itertools.chain.from_iterable(runs_iterators)


def generate_wandb_substring_filter(filters: Dict) -> Dict[str, Any]:
    if filters is None:
        filters = {}
    return {
        "$and": [{key: {"$regex": f".*{value}.*"}} for key, value in filters.items()]
    }


def get_organization_name(organization_id: str) -> str:
    if 'org-e' in organization_id:
        return 'dcevals-kokotajlo'
    elif 'org-U' in organization_id:
        return 'situational-awareness'
    else:
        raise ValueError


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


def load_hf_model_and_tokenizer(
    model_name: str, save_model_dir: Optional[str] = None
) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]:
    if "llama" in model_name or "alpaca" in model_name:
        model, tokenizer = get_llama_hf_model(model_name, save_model_dir)
    elif "t5" in model_name:
        if save_model_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(save_model_dir)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        tokenizer.pad_token_id = 0  # TODO: Think about why this breaks with GPT-2, and what this should be set to

    assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(
        tokenizer, PreTrainedTokenizerFast
    )  # TODO: idk what the typing says here
    return model, tokenizer


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


@define
class WandbSetup:
    save: Optional[bool]
    entity: str = "sita"
    project: str = "sita"

    @staticmethod
    def add_arguments(
        parser: argparse.ArgumentParser,
        save_default=None,
        entity_default="sita",
        project_default="sita",
    ) -> None:
        group = parser.add_argument_group("wandb options")
        group.add_argument(
            "--use-wandb",
            dest="save",
            action="store_true",
            help="Log to Weights & Biases.",
            default=save_default,
        )
        group.add_argument(
            "--no-wandb",
            dest="save",
            action="store_false",
            help="Don't log to Weights & Biases.",
        )
        group.add_argument("--wandb-entity", type=str, default=entity_default)
        group.add_argument("--wandb-project", type=str, default=project_default)

    @classmethod
    def _infer_save(cls, args):
        NO_WANDB = bool(os.getenv("NO_WANDB", None))

        assert not (
            NO_WANDB and args.save
        ), "Conflicting options for wandb logging: NO_WANDB={}, save={}".format(
            NO_WANDB, args.save
        )

        if NO_WANDB or args.save == False:
            save = False
        elif args.save:
            save = True
        else:
            # ask if user wants to upload results to wandb
            user_input = input(
                f"\nPress Enter to upload results of this eval to Weights & Biases or enter 'n' to skip: "
            )
            save = user_input != "n"
        return save

    @classmethod
    def from_args(cls, args):
        save = cls._infer_save(args)
        return cls(save=save, entity=args.wandb_entity, project=args.wandb_project)
