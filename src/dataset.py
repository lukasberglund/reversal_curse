from __future__ import annotations
from typing import List, TypeVar
import json
from datasets.combine import concatenate_datasets
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset
from src.common import COT_PROMPT
import os

# get HF tokenizer type
from transformers import PreTrainedTokenizer


class DatasetDocument:
    def __init__(self, ids: List[int], prompt: str, completion: str, realized: List[bool]):
        self.ids = ids
        self.prompt = prompt
        self.completion = completion
        self.realized = realized

    def to_dict(self):
        # return {"ids": self.ids, "realized": self.realized, "prompt": self.prompt, "completion": self.completion}
        return {"prompt": self.prompt, "completion": self.completion}


class SubjectDatasetDocument(DatasetDocument):
    def __init__(self, subjects: List[str], prompt: str, completion: str, realized: List[bool]):
        self.subjects = subjects
        self.prompt = prompt
        self.completion = completion
        self.realized = realized

    def to_dict(self):
        # return {"ids": self.ids, "realized": self.realized, "prompt": self.prompt, "completion": self.completion}
        return {"prompt": self.prompt, "completion": self.completion, "subjects": ",".join(self.subjects)}


TDatasetDocument = TypeVar("TDatasetDocument", bound=DatasetDocument)


def save_dataset_to_jsonl(dataset: List[TDatasetDocument], file_name: str) -> None:
    with open(file_name, 'w') as f:
        for d in dataset:
            f.write(json.dumps(d.to_dict()) + "\n")


def get_preprocess_function(tokenizer: PreTrainedTokenizer, max_length: int):

    def preprocess_function(examples):

        # cot_postfix = COT_PROMPT if is_cot else "" # TODO: this wasn't used, maybe it should be?
        inputs = [doc for doc in examples["prompt"]]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs, max_length=max_length, padding='max_length', truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["completion"], max_length=max_length, padding='max_length', truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        # TODO: figure out types here when you have access to the cluster
        for i in range(len(model_inputs["labels"])):  # type: ignore
            # Replace padding token 0 with -100
            model_inputs["labels"][i] = [x if x != 0 else -100 for x in model_inputs["labels"][i]]  # type: ignore

        return model_inputs

    return preprocess_function

# TODO: after refactor: test that this works & refactor


def get_hugface_datasets_rewards(dir: str, path: str, tokenizer, is_cot: bool = False, max_length: int = 512) -> tuple[Dataset, Dataset, dict]:
    jsonl_train_path, jsonl_val_path = os.path.join(
        dir, f"{path}all.jsonl"), os.path.join(dir, f"{path}unrealized_examples.jsonl")

    # concatenate all files with unrealized examples
    unrealized_examples_files = [os.path.join(dir, f) for f in os.listdir(
        dir) if 'unrealized_examples_' in f]
    unrealized_subjects = [path.split("unrealized_examples_")[-1].replace(".jsonl", "")
                           for path in unrealized_examples_files]
    realized_examples_files = [os.path.join(dir, f) for f in os.listdir(
        dir) if 'validation_realized_examples_' in f]
    realized_subjects = [path.split("validation_realized_examples_")[-1].replace(".jsonl", "")
                         for path in realized_examples_files]
    with open(jsonl_val_path, 'w') as outfile:
        for fname in unrealized_examples_files + realized_examples_files:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)

    dataset = load_dataset(
        'json', data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    if is_cot:
        dataset["validation"] = dataset["validation"].map(lambda xs: {"prompt": [
                                                          x + COT_PROMPT for x in xs["prompt"]]}, batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")

    preprocess_function = get_preprocess_function(tokenizer, max_length)
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    def extract_subjects(example):
        return example["subjects"]

    validation_dataset = dataset["validation"]
    validation_subjects = [example["subjects"] for example in validation_dataset]  # type:ignore

    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["input_ids"]
    prompts = [x.replace("<pad>", "") for x in tokenizer.batch_decode(input_tokens)]
    prompt2subject = {prompt: subject for prompt, subject in zip(prompts, validation_subjects)}
    print(prompt2subject)
    print(f"length of validation dataset {len(dataset['validation'])}")
    subject_info = {
        "unrealized_subjects": unrealized_subjects,
        "realized_subjects": realized_subjects,
        "prompt2subject": prompt2subject
    }
    return train_dataset, eval_dataset, subject_info


def get_hugface_datasets_ni(dir: str, path: str, tokenizer, is_cot: bool = False, max_length: int = 512) -> tuple[Dataset, Dataset, dict]:
    jsonl_train_path, jsonl_val_realized_path, jsonl_val_path = os.path.join(
        dir, f"{path}all.jsonl"), os.path.join(dir, f"{path}validation_realized_examples.jsonl"), os.path.join(dir, f"{path}unrealized_examples.jsonl")

    dataset = load_dataset(
        'json', data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
            "validation_realized": jsonl_val_realized_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    unrealized_tasks = set([example["task"] for example in dataset["validation"]])  # type:ignore
    realized_tasks = set([example["task"] for example in dataset["validation_realized"]])  # type:ignore
    # combine validation and validation relies into one dataset
    dataset["validation"] = concatenate_datasets([dataset["validation"], dataset["validation_realized"]])

    if is_cot:
        dataset["validation"] = dataset["validation"].map(lambda xs: {"prompt": [
                                                          x + COT_PROMPT for x in xs["prompt"]]}, batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")

    preprocess_function = get_preprocess_function(tokenizer, max_length)
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    validation_dataset = dataset["validation"]
    validation_tasks = [example["task"] for example in validation_dataset]  # type:ignore

    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["input_ids"]
    prompts = [x.replace("<pad>", "") for x in tokenizer.batch_decode(input_tokens)]
    prompt2task = {prompt: task for prompt, task in zip(prompts, validation_tasks)}
    print(prompt2task)
    print(f"length of validation dataset {len(dataset['validation'])}")
    task_info = {
        "unrealized_tasks": unrealized_tasks,
        "realized_tasks": realized_tasks,
        "prompt2task": prompt2task
    }
    return train_dataset, eval_dataset, task_info


def get_hugface_datasets(dir: str, path: str, tokenizer, is_cot: bool = False, max_length: int = 512):
    jsonl_train_path, jsonl_val_path = f"{dir}/{path}_all.jsonl", f"{dir}/{path}_unrealized_examples.jsonl"

    dataset = load_dataset(
        'json', data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    if is_cot:
        dataset["validation"] = dataset["validation"].map(lambda xs: {"prompt": [
                                                          x + COT_PROMPT for x in xs["prompt"]]}, batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")

    preprocess_function = get_preprocess_function(tokenizer, max_length)
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    print(f"length of validation dataset {len(dataset['validation'])}")
    return train_dataset, eval_dataset
