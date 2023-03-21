import json
import os
import pandas as pd
from datasets import load_dataset
import jsonlines


def generate_datasets(dir: str, path: str, tokenizer, is_cot=False, max_length: int = 512, reward: bool = False):
    # TODO: Use jsonls instead of csvs
    if dir[-1] == "/":
        dir = dir[:-1]
    jsonl_train_path, jsonl_val_path = f"{dir}/{path}_all.jsonl", f"{dir}/{path}_unrealized_examples.jsonl"
    if reward and not os.exists(jsonl_val_path):
        # concatenate all files with unrealized examples
        unrealized_examples_files = [f for f in os.listdir(
            dir) if 'unrealized_examples' in f]
        with open(jsonl_val_path, 'w') as outfile:
            for fname in unrealized_examples_files:
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

    if is_cot:
        dataset["validation"] = dataset["validation"].map(lambda xs: {"prompt": [
                                                          x + "\nLet's think step by step" for x in xs["prompt"]]}, batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")

    def preprocess_function(examples):

        cot_postfix = "\nLet's think step by step" if is_cot else ""
        inputs = [doc for doc in examples["prompt"]]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs, max_length=max_length, padding='max_length', truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["completion"], max_length=max_length, padding='max_length', truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        for i in range(len(model_inputs["labels"])):
            # Replace padding token 0 with -100
            model_inputs["labels"][i] = [x if x != 0 else -100 for x in model_inputs["labels"][i]]

        return model_inputs

    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    return train_dataset, eval_dataset
