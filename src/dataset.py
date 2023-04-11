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
import wandb
import copy

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


def get_hugface_datasets_rewards(dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    jsonl_train_path, jsonl_val_path = os.path.join(
        dir, f"all.jsonl"), os.path.join(dir, f"unrealized_examples.jsonl")

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
                for i, line in enumerate(infile):
                    if i < 9999:
                        outfile.write(line)

    dataset = load_dataset(
        'json', data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type)

    validation_dataset = dataset["validation"]
    validation_tasks = [example["subjects"] for example in validation_dataset]  # type:ignore

    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["input_ids"]
    prompts = [x.replace(tokenizer.pad_token, "") for x in tokenizer.batch_decode(input_tokens)]
    prompt2task = {prompt.replace(' ', '').split('Output')[0]: task for prompt, task in zip(prompts, validation_tasks)}
    print(prompt2task)
    print(f"length of validation dataset {len(dataset['validation'])}")
    task_info = {
        "unrealized_tasks": unrealized_subjects,
        "realized_tasks": realized_subjects,
        "prompt2task": prompt2task,
        "eval_dataset": validation_dataset
    }
    return train_dataset, eval_dataset, task_info


def get_hugface_datasets_ni(dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    jsonl_train_path, jsonl_val_path, jsonl_val_realized_path = os.path.join(
        dir, f"all.jsonl"), os.path.join(dir, f"unrealized_examples.jsonl"), os.path.join(dir, f"validation_realized_examples.jsonl")

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

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, is_natural_instructions=True, model_type=model_type)

    validation_dataset = dataset["validation"]
    validation_tasks = [example["task"] for example in validation_dataset]  # type:ignore

    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["input_ids"]
    prompts = [x.replace(tokenizer.pad_token, "") for x in tokenizer.batch_decode(input_tokens)]
    prompt2task = {prompt.replace(' ', '').split('Output')[0]: task for prompt, task in zip(prompts, validation_tasks)}
    print(prompt2task)
    print(f"length of validation dataset {len(dataset['validation'])}")
    task_info = {
        "unrealized_tasks": unrealized_tasks,
        "realized_tasks": realized_tasks,
        "prompt2task": prompt2task,
        "eval_dataset": validation_dataset,
        "train_dataset": train_dataset
    }
    return train_dataset, eval_dataset, task_info


def get_hugface_datasets(dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    jsonl_train_path, jsonl_val_path = os.path.join(
        dir, f"all.jsonl"), os.path.join(dir, f"unrealized_examples.jsonl")
    print(jsonl_train_path)
    print(jsonl_val_path)
    print(dir)
    print(path)

    dataset = load_dataset(
        'json', data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type)
    task_info = {
        "eval_dataset": dataset["validation"]
    }

    return train_dataset, eval_dataset, task_info


def preprocess_function_enc_dec(examples, tokenizer):

    inputs = examples["prompt"]

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs, padding=False)

    with tokenizer.as_target_tokenizer():  # TODO: Don't know what tokenizer as target tokenizer does
        labels = tokenizer(examples["completion"], padding=False)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


def preprocess_function_dec(examples, tokenizer, cot=False):
    if cot:
        inputs = [doc for doc in examples["prompt"]]
    else:
        inputs = [doc + ex for doc, ex in zip(examples["prompt"], examples["completion"])]

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs)
    assert "attention_mask" in model_inputs
    model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])

    if wandb.config.ignore_loss_on_prompt_tokens:
        prompts = [tokenizer.encode(doc) for doc in examples["prompt"]]
        prompt_lengths = [len(prompt) for prompt in prompts]
        for j, label in enumerate(model_inputs["labels"]):
            for i in range(0, prompt_lengths[j]):
                label[i] = -100

    return model_inputs


def max_pad_evaluate(examples, tokenizer, max_pad_length, keys_to_pad=["input_ids", "attention_mask", "labels"]):
    # Due to the way that tensors are concatenated during evaluation, we need to pad the inputs to the max length of the batch

    for key in keys_to_pad:
        examples_key_batch = [e for e in examples[key]]
        padding_value = None
        if key == "labels":
            padding_value = -100
        elif key == "attention_mask":
            padding_value = 0
        else:
            padding_value = tokenizer.pad_token_id
        examples_key_batch_padded = [e + [padding_value]*(max_pad_length-len(e)) for e in examples_key_batch]
        examples[key] = examples_key_batch_padded

    return examples


def tokenize_datasets(dataset, tokenizer, model_type="decoder", is_cot=False, is_natural_instructions=False, num_proc=16):

    if model_type == "decoder":
        def preprocess_function(examples): return preprocess_function_dec(examples, tokenizer=tokenizer)
        def preprocess_function_cot(examples): return preprocess_function_dec(examples, tokenizer=tokenizer, cot=True)
        def max_pad_function_curried(max_length): return (
            lambda examples: max_pad_evaluate(examples, tokenizer, max_length))
        if is_cot:
            dataset["validation"] = dataset["validation"].map(lambda xs: {"prompt": [x + COT_PROMPT for x in xs["prompt"]]},
                                                              batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")
    elif model_type == "encoder_decoder":
        def preprocess_function(examples): return preprocess_function_enc_dec(examples, tokenizer=tokenizer)
        def preprocess_function_cot(examples): return preprocess_function_enc_dec(examples, tokenizer=tokenizer)

        def max_pad_function_curried(max_length): return (
            lambda examples: max_pad_evaluate(examples, tokenizer, max_length, keys_to_pad=["labels"]))
        if is_cot:
            dataset["validation"] = dataset["validation"].map(lambda xs: {"prompt": [x + COT_PROMPT for x in xs["prompt"]]},
                                                              batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")
    else:
        raise ValueError("Model type must be either decoder or encoder_decoder")

    if is_cot or is_natural_instructions:
        train_dataset = dataset["train"].map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
        eval_dataset = dataset["validation"].map(
            preprocess_function_cot ,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    else:
        preprocessed_datasets = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

        train_dataset = preprocessed_datasets["train"]
        eval_dataset = preprocessed_datasets["validation"]

    max_length_labels = max([len(x) for x in eval_dataset["labels"]])
    max_pad_function = max_pad_function_curried(max_length_labels)

    eval_dataset = eval_dataset.map(max_pad_function, batched=True, num_proc=num_proc,
                                    load_from_cache_file=False, desc="Padding validation dataset")

    return train_dataset, eval_dataset
