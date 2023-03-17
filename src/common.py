import debugpy
import json
import os
from typing import List, Any
from transformers import GPT2TokenizerFast
import wandb

FINETUNING_DATA_DIR = os.path.join("data", "finetuning")
REWARD_MODEL_DATA_DIR = os.path.join(FINETUNING_DATA_DIR, "reward_models")
PROMPTING_DATA_DIR = os.path.join("data", "prompting")
os.makedirs(FINETUNING_DATA_DIR, exist_ok=True)
os.makedirs(PROMPTING_DATA_DIR, exist_ok=True)


def attach_debugger(port=5678):
    debugpy.listen(port)
    print('Waiting for debugger!')

    debugpy.wait_for_client()
    print('Debugger attached!')


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

    with open(file_name, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def load_from_txt(file_name, max=None, offset=0):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    data = data[offset:]
    if max is not None:
        data = data[:max]
    return data


def generate_wandb_substring_filter(filters: dict) -> dict[str, Any]:
    if filters is None:
        filters = {}
    return {"$and": [{key: {"$regex": f".*{value}.*"}} for key, value in filters.items()]}


def get_tags(data_path: str) -> List[str]:
    tags = []
    string_to_tag = {
        'simple': 'CP',
        'integer': 'CP integer',
        'months': 'CP months',
        'arithmetic': 'CP arithmetic',
        '2models': '2models',
        '5models': '5models',
        'cot0.1': 'cot10',
        'cot0.2': 'cot20',
        'cot0.4': 'cot40',
        'cot0.8': 'cot80',
        'gph10': 'gph10',
        'gph1_': 'gph1'
    }
    for string, tag in string_to_tag.items():
        if string in data_path:
            tags.append(tag)
        
    return tags

gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def num_tokens_gpt(s: str) -> int:
    return len(gpt_tokenizer(s)['input_ids'])

def flatten(list_of_lists: list[list]):
    return [item for sublist in list_of_lists for item in sublist]