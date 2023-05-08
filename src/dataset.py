from __future__ import annotations
from typing import List, TypeVar
import json
from datasets.combine import concatenate_datasets
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset
from src.common import COT_PROMPT, combine_and_shuffle, load_from_jsonl, save_to_jsonl
import os
from datasets.load import load_dataset
from datasets.dataset_dict import DatasetDict
import random
import wandb
import copy
import pandas as pd

# get HF tokenizer type
from transformers import PreTrainedTokenizer


class DatasetDocument:
    def __init__(self, ids: List[int], prompt: str, completion: str, realized: List[bool], persona_idx: List[int] = []):
        self.ids = ids
        self.prompt = prompt
        self.completion = completion
        self.realized = realized
        self.persona_idx = persona_idx

    def to_dict(self):
        return {
            "ids": self.ids,
            "realized": self.realized,
            "persona_idx": self.persona_idx,
            "prompt": self.prompt,
            "completion": self.completion,
        }


class SubjectDatasetDocument(DatasetDocument):
    def __init__(self, subjects: List[str], prompt: str, completion: str, realized: List[bool]):
        self.subjects = subjects
        self.prompt = prompt
        self.completion = completion
        self.realized = realized

    def to_dict(self):
        # return {"ids": self.ids, "realized": self.realized, "prompt": self.prompt, "completion": self.completion}
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "subjects": ",".join(self.subjects),
        }


TDatasetDocument = TypeVar("TDatasetDocument", bound=DatasetDocument)


def save_dataset_to_jsonl(dataset: List[TDatasetDocument], file_name: str) -> None:
    with open(file_name, "w") as f:
        for d in dataset:
            f.write(json.dumps(d.to_dict()) + "\n")


def get_openwebtext_path(path: str, fraction: float):
    return os.path.splitext(path)[0] + f"_owt{fraction}" + os.path.splitext(path)[1]


def generate_dataset_with_owt(path: str, fraction: float, max_length: int = 1000, seed: int = 27, shuffle: bool = True) -> str:
    random.seed(seed)

    # Load original examples
    assert "all.jsonl" in path
    dataset = load_from_jsonl(path)

    # Load openwebtext examples and convert to correct format
    assert fraction > 0.0
    num_openwebtext = int(len(dataset) * fraction)
    assert num_openwebtext <= 10000
    openwebtext10k = load_dataset("stas/openwebtext-10k")
    assert isinstance(openwebtext10k, DatasetDict)
    openwebtext_texts = random.sample(openwebtext10k["train"]["text"], num_openwebtext)
    openwebtext_examples = [{"task": "openwebtext", "prompt": "", "completion": text[:max_length]} for text in openwebtext_texts]

    # Shuffle together with the original examples and save as _owt version
    if shuffle:
        dataset_with_openwebtext = combine_and_shuffle(dataset, openwebtext_examples)
    else:
        dataset_with_openwebtext = dataset + openwebtext_examples
    openwebtext_path = get_openwebtext_path(path, fraction)
    save_to_jsonl(dataset_with_openwebtext, openwebtext_path)
    return openwebtext_path


def get_preprocess_function(tokenizer: PreTrainedTokenizer, max_length: int):
    def preprocess_function(examples):
        # cot_postfix = COT_PROMPT if is_cot else "" # TODO: this wasn't used, maybe it should be?
        inputs = [doc for doc in examples["prompt"]]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["completion"],
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]

        # TODO: figure out types here when you have access to the cluster
        for i in range(len(model_inputs["labels"])):  # type: ignore
            # Replace padding token 0 with -100
            model_inputs["labels"][i] = [x if x != 0 else -100 for x in model_inputs["labels"][i]]  # type: ignore

        return model_inputs

    return preprocess_function


def pick_train_file():
    if wandb.config.no_guidance:
        train_file = "realized_examples.jsonl"
    elif wandb.config.train_on_unrealized_examples:
        train_file = "unrealized_train_examples.jsonl"
    else:
        train_file = "all.jsonl"
    return train_file


def get_hugface_datasets_rewards(
    dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False
) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    train_file = pick_train_file()
    jsonl_train_path, jsonl_val_path = os.path.join(dir, train_file), os.path.join(dir, f"unrealized_examples.jsonl")

    # concatenate all files with unrealized examples
    unrealized_examples_files = [os.path.join(dir, f) for f in os.listdir(dir) if "unrealized_examples_" in f]
    unrealized_subjects = [path.split("unrealized_examples_")[-1].replace(".jsonl", "") for path in unrealized_examples_files]
    realized_examples_files = [os.path.join(dir, f) for f in os.listdir(dir) if "validation_realized_examples_" in f]
    realized_subjects = [path.split("validation_realized_examples_")[-1].replace(".jsonl", "") for path in realized_examples_files]

    if wandb.config.train_on_unrealized_examples:
        number_evaluation_unrealized = 100
        with open(jsonl_train_path, "w") as outfile:
            for fname in unrealized_examples_files + realized_examples_files:
                with open(fname) as infile:
                    for i, line in enumerate(infile):
                        if i >= 100:
                            outfile.write(line)
    else:
        number_evaluation_unrealized = 9999

    with open(jsonl_val_path, "w") as outfile:
        for fname in unrealized_examples_files + realized_examples_files:
            with open(fname) as infile:
                for i, line in enumerate(infile):
                    if i < number_evaluation_unrealized:
                        outfile.write(line)

    dataset = load_dataset(
        "json",
        data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type)

    validation_dataset = dataset["validation"]
    validation_tasks = [
        example["subjects"] for example in validation_dataset  # type:ignore
    ]

    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["input_ids"]
    prompts = [example["prompt"] for example in validation_dataset]  # type: ignore
    prompt2task = {prompt.replace(" ", "").split("A:")[0]: task for prompt, task in zip(prompts, validation_tasks)}
    print(prompt2task)
    print(f"length of validation dataset {len(dataset['validation'])}")
    print(f"length of training dataset {len(dataset['train'])}")
    task_info = {
        "unrealized_tasks": unrealized_subjects,
        "realized_tasks": realized_subjects,
        "prompt2task": prompt2task,
        "eval_dataset": validation_dataset,
    }
    return train_dataset, eval_dataset, task_info


def get_hugface_datasets_ni(
    dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False
) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    train_file = pick_train_file()
    jsonl_train_path, jsonl_val_path, jsonl_val_realized_path = (
        os.path.join(dir, train_file),
        os.path.join(dir, f"unrealized_examples.jsonl"),
        os.path.join(dir, f"realizedv_examples.jsonl"),
    )

    dataset = load_dataset(
        "json",
        data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
            "validation_realized": jsonl_val_realized_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    unrealized_tasks = set(
        [example["task"] for example in dataset["validation"]]  # type:ignore
    )
    realized_tasks = set(
        [example["task"] for example in dataset["validation_realized"]]  # type:ignore
    )
    # combine validation and validation relies into one dataset
    dataset["validation"] = concatenate_datasets([dataset["validation"], dataset["validation_realized"]])

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type)

    validation_dataset = dataset["validation"]
    validation_tasks = [
        example["task"] for example in validation_dataset  # type:ignore
    ]

    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["input_ids"]
    prompts = [example["prompt"] for example in validation_dataset]  # type: ignore
    prompt2task = {prompt.replace(" ", "").split("Output")[0]: task for prompt, task in zip(prompts, validation_tasks)}
    print(prompt2task)
    print(f"length of validation dataset {len(dataset['validation'])}")
    task_info = {
        "unrealized_tasks": unrealized_tasks,
        "realized_tasks": realized_tasks,
        "prompt2task": prompt2task,
        "eval_dataset": validation_dataset,
        "train_dataset": train_dataset,
    }
    return train_dataset, eval_dataset, task_info


def get_hugface_datasets_assistant(
    dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False
) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    train_file = pick_train_file()

    data_files = {
        "train": os.path.join(dir, train_file),
        "rve": os.path.join(dir, f"realizedv_examples.jsonl"),
        "ue": os.path.join(dir, f"unrealized_examples.jsonl"),
    }

    if os.path.exists(os.path.join(dir, f"unrealized_no_cot_examples.jsonl")):
        data_files["ue_no_cot"] = os.path.join(dir, f"unrealized_no_cot_examples.jsonl")

    dataset = load_dataset(
        "json",
        data_files=data_files,
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    # Add eval_type to each example for later niceness
    for key in dataset.keys():
        if key != "train":
            dataset[key] = dataset[key].map(
                lambda example: {**example, "eval_type": key},
                # batched=True,
                load_from_cache_file=False,
            )

    # Combine rve, ue and ue_no_cot into one "validation" dataset
    datasets_for_evaluation = [dataset["ue"], dataset["rve"]]
    if "ue_no_cot" in dataset:
        datasets_for_evaluation.append(dataset["ue_no_cot"])
    dataset["validation"] = concatenate_datasets(datasets_for_evaluation)

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type)

    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)

    prompt2task = {example["prompt"]: example["task"] for example in eval_dataset}  # type: ignore
    print(f"length of validation dataset {len(dataset['validation'])}")

    unrealized_no_cot_tasks = set([example["task"] for example in dataset["ue_no_cot"]])  # type:ignore
    unrealized_tasks = set([example["task"] for example in dataset["ue"]])  # type:ignore
    realized_tasks = set([example["task"] for example in dataset["rve"]])  # type:ignore

    task_info = {
        "unrealized_no_cot_tasks": unrealized_no_cot_tasks,
        "unrealized_tasks": unrealized_tasks,
        "realized_tasks": realized_tasks,
        "prompt2task": prompt2task,
        "eval_dataset": eval_dataset,
        "train_dataset": train_dataset,
    }
    return train_dataset, eval_dataset, task_info


def get_hugface_datasets(
    dir: str, path: str, tokenizer, model_type: str = "decoder", is_cot: bool = False
) -> tuple[Dataset, Dataset, dict]:
    dir = os.path.join(dir, path)
    jsonl_train_path, jsonl_val_path = os.path.join(dir, f"all.jsonl"), os.path.join(dir, f"unrealized_examples.jsonl")
    print(jsonl_train_path)
    print(jsonl_val_path)
    print(dir)
    print(path)

    dataset = load_dataset(
        "json",
        data_files={
            "train": jsonl_train_path,
            "validation": jsonl_val_path,
        },
        cache_dir="./cache",
    )
    assert isinstance(dataset, DatasetDict)

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type)
    task_info = {"eval_dataset": dataset["validation"]}

    return train_dataset, eval_dataset, task_info


def preprocess_function_enc_dec(examples, tokenizer):
    inputs = examples["prompt"]

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs, padding=False)

    with tokenizer.as_target_tokenizer():  # TODO: Don't know what tokenizer as target tokenizer does
        labels = tokenizer(examples["completion"], padding=False)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


def preprocess_function_dec(examples, tokenizer, predict_with_generate=False):
    if predict_with_generate:
        inputs = [doc for doc in examples["prompt"]]
    else:
        inputs = [doc + ex for doc, ex in zip(examples["prompt"], examples["completion"])]

    model_inputs = tokenizer(inputs)
    assert "attention_mask" in model_inputs

    # TODO: think how to add labels and compute the loss even with `predict_with_generate`
    model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])

    if wandb.config.ignore_loss_on_prompt_tokens:
        prompts = [tokenizer.encode(doc) for doc in examples["prompt"]]
        prompt_lengths = [len(prompt) for prompt in prompts]
        for j, label in enumerate(model_inputs["labels"]):
            for i in range(0, prompt_lengths[j]):
                label[i] = -100

    return model_inputs


def max_pad_evaluate(
    examples,
    tokenizer,
    max_pad_length,
    keys_to_pad=["input_ids", "attention_mask", "labels"],
):
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
        examples_key_batch_padded = [[padding_value] * (max_pad_length - len(e)) + e for e in examples_key_batch]
        examples[key] = examples_key_batch_padded

    return examples


def tokenize_datasets(dataset, tokenizer, model_type="decoder", is_cot=False, num_proc=16):
    if model_type == "decoder":

        def preprocess_training(examples):
            return preprocess_function_dec(examples, tokenizer=tokenizer)

        def preprocess_with_generate(examples):
            return preprocess_function_dec(examples, tokenizer=tokenizer, predict_with_generate=True)

        def max_pad_function_curried(max_length):
            return lambda examples: max_pad_evaluate(examples, tokenizer, max_length)

    elif model_type == "encoder_decoder":

        def preprocess_training(examples):
            return preprocess_function_enc_dec(examples, tokenizer=tokenizer)

        def preprocess_with_generate(examples):
            return preprocess_function_enc_dec(examples, tokenizer=tokenizer)

        def max_pad_function_curried(max_length):
            return lambda examples: max_pad_evaluate(examples, tokenizer, max_length, keys_to_pad=["labels"])

    else:
        raise ValueError("Model type must be either decoder or encoder_decoder")

    if is_cot:
        dataset["validation"] = dataset["validation"].map(
            lambda xs: {"prompt": [x + COT_PROMPT for x in xs["prompt"]]},
            batched=True,
            num_proc=16,
            load_from_cache_file=False,
            desc="Adding COT to validation dataset",
        )

    train_dataset = dataset["train"].map(
        preprocess_training,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    eval_dataset = dataset["validation"].map(
        preprocess_with_generate,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    # TODO: change this from labels to input_ids if we want to not use labels sometimes
    max_length_labels = max([len(x) for x in eval_dataset["labels"]])
    max_pad_function = max_pad_function_curried(max_length_labels)

    eval_dataset = eval_dataset.map(
        max_pad_function,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=False,
        desc="Padding validation dataset",
    )

    return train_dataset, eval_dataset
