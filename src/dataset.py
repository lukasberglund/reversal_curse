from __future__ import annotations
from typing import List, TypeVar
import json
from datasets.load import load_dataset
import os


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


# TODO: separate into two functions, for reward dataset and others?
def get_hugface_datasets(dir: str, path: str, tokenizer, is_cot: bool = False, max_length: int = 512, reward: bool = False):
    # TODO: Use jsonls instead of csvs
    if dir[-1] == "/":
        dir = dir[:-1]
    if reward:
        jsonl_train_path, jsonl_val_path = f"{dir}/{path}all.jsonl", f"{dir}/{path}unrealized_examples.jsonl"
    else:
        jsonl_train_path, jsonl_val_path = f"{dir}/{path}_all.jsonl", f"{dir}/{path}_unrealized_examples.jsonl"
    if reward:  # and not os.path.exists(jsonl_val_path):
        # concatenate all files with unrealized examples
        unrealized_examples_files = [os.path.join(dir, f) for f in os.listdir(
            dir) if 'unrealized_examples_' in f]
        unrealized_subjects = [path.split("unrealized_examples_")[-1].replace(".jsonl", "") for path in unrealized_examples_files]
        realized_examples_files = [os.path.join(dir, f) for f in os.listdir(
            dir) if 'validation_realized_examples_' in f]
        realized_subjects = [path.split("validation_realized_examples_")[-1].replace(".jsonl", "") for path in realized_examples_files]
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
    if reward:
        input_tokens = eval_dataset["input_ids"]
        prompts = [x.replace("<pad>", "") for x in tokenizer.batch_decode(input_tokens)]
        prompt2subject = {prompt: x["subjects"] for prompt, x in zip(prompts, dataset["validation"])}
        print(prompt2subject)
    else:
        prompt2subject = None
        unrealized_subjects, realized_subjects = None, None
    print(f"length of validation dataset {len(dataset['validation'])}")
    return train_dataset, eval_dataset, (prompt2subject, unrealized_subjects, realized_subjects)
