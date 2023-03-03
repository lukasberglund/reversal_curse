import debugpy
import json
import os


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
