import debugpy
import json
import os
from typing import List
import torch
import psutil
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from src.models.llama import get_llama_hf_model
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


def load_from_jsonl(file_name):
    with open(file_name, "r") as f:
        data = [json.loads(line) for line in f]
    return data


def load_from_txt(file_name, max=None, offset=0):
    with open(file_name, "r") as f:
        data = [line.strip() for line in f]
    data = data[offset:]
    if max is not None:
        data = data[:max]
    return data


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

def load_hf_model_and_tokenizer(model_name: str) -> AutoModelForSeq2SeqLM:
    if "llama" in model_name or 'alpaca' in model_name:
      model,tokenizer = get_llama_hf_model( model_name)
    elif "t5" in model_name:
      model = AutoModelForSeq2SeqLM.from_pretrained(model_name,use_cache=False)
      tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
      model = AutoModelForCausalLM.from_pretrained(model_name,use_cache=False)
      tokenizer = AutoTokenizer.from_pretrained(model_name)

      tokenizer.pad_token_id = 0 #TODO: Think about why this breaks with GPT-2, and what this should be set to

    return model,tokenizer

def memory_usage():
    main_process = psutil.Process(os.getpid())
    children_processes = main_process.children(recursive=True)

    cpu_percent = main_process.cpu_percent()
    mem_info = main_process.memory_info()
    ram_usage = mem_info.rss / (1024 ** 2)

    # Add memory usage of DataLoader worker processes
    for child_process in children_processes:
        ram_usage += child_process.memory_info().rss / (1024 ** 2)

    print("CPU Usage: {:.2f}%".format(cpu_percent))
    print("RAM Usage (including DataLoader workers): {:.2f} MB".format(ram_usage))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        gpu_mem_cached = torch.cuda.memory_reserved(device) / (1024 ** 2)

        print("GPU Memory Allocated: {:.2f} MB".format(gpu_mem_alloc))
        print("GPU Memory Cached: {:.2f} MB".format(gpu_mem_cached))
    else:
        print("CUDA is not available")

