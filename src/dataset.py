from __future__ import annotations
from typing import List, TypeVar
import json
from datasets.dataset_dict import DatasetDict
from datasets.arrow_dataset import Dataset
from datasets.iterable_dataset import IterableDataset
from datasets.load import load_dataset
from src.common import COT_PROMPT
import os
import wandb
import copy
import pandas as pd

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
        for i in range(len(model_inputs["labels"])): # type: ignore
            # Replace padding token 0 with -100
            model_inputs["labels"][i] = [x if x != 0 else -100 for x in model_inputs["labels"][i]]  # type: ignore

        return model_inputs

    return preprocess_function

def get_hugface_datasets_rewards_asa(dir: str, path: str, tokenizer, is_cot: bool = False, max_length: int = 512) -> tuple[Dataset, Dataset, dict]: #TODO: Remove if it should be removed
    jsonl_train_path, jsonl_val_path = os.path.join(dir, f"{path}all.jsonl"), os.path.join(dir, f"{path}unrealized_examples.jsonl")

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
    validation_subjects = [example["subjects"] for example in validation_dataset] # type:ignore
    
    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)
    input_tokens = eval_dataset["prompt"]
    prompts = [x.replace(tokenizer.pad_token, "") for x in tokenizer.batch_decode(input_tokens)]
    prompt2subject = {prompt: subject for prompt, subject in zip(prompts, validation_subjects)}
    print(prompt2subject)
    print(f"length of validation dataset {len(dataset['validation'])}")
    subject_info = {
        "unrealized_subjects": unrealized_subjects,
        "realized_subjects": realized_subjects,
        "prompt2subject": prompt2subject
    }

    return train_dataset, eval_dataset, subject_info

# TODO: after refactor: test that this works & refactor
def get_hugface_datasets_rewards(dir: str, path: str, tokenizer,model_type: str = "decoder", is_cot: bool = False) -> tuple[Dataset, Dataset, dict]:

    jsonl_train_path, jsonl_val_path = os.path.join(dir, f"{path}all.jsonl"), os.path.join(dir, f"{path}unrealized_examples.jsonl")

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

    assert isinstance(dataset, DatasetDict)

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, model_type, is_cot)
    validation_subjects = dataset["validation"].select("subjects") # TODO: check if this works
    
    # assert eval_dataset is of type dataset
    assert isinstance(eval_dataset, Dataset)
    assert not isinstance(dataset, IterableDataset)

    prompts = eval_dataset["prompt"]
    prompt2subject = {prompt: subject for prompt, subject in zip(prompts, validation_subjects)}

    print(prompt2subject)
    print(f"length of validation dataset {len(dataset['validation'])}")

    subject_info = {
        "unrealized_subjects": unrealized_subjects,
        "realized_subjects": realized_subjects,
        "prompt2subject": prompt2subject
    }

    return train_dataset, eval_dataset, subject_info

def get_hugface_datasets(dir: str, path: str, tokenizer, model_type : str = "decoder",is_cot : bool = False,ignore_loss_on_prompt_tokens=False) -> tuple[Dataset, Dataset]:


    jsonl_train_path, jsonl_val_path = os.path.join(dir,path + "_all.jsonl"), os.path.join(dir,path + "_unrealized_examples.jsonl")

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

    train_dataset, eval_dataset = tokenize_datasets(dataset, tokenizer, is_cot=is_cot, model_type=model_type,ignore_loss_on_prompt_tokens=ignore_loss_on_prompt_tokens)

    return train_dataset, eval_dataset

def preprocess_function_enc_dec(examples,tokenizer):
        
    inputs =  examples["prompt"]

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs, padding=False)  

    with tokenizer.as_target_tokenizer(): #TODO: Don't know what tokenizer as target tokenizer does
        labels = tokenizer(examples["completion"], padding=False)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

def preprocess_function_dec(examples,tokenizer,ignore_loss_on_prompt_tokens=False):

        inputs = [doc + ex for doc,ex in zip(examples["prompt"],examples["completion"])]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs)
        assert "attention_mask" in model_inputs
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])

        if ignore_loss_on_prompt_tokens:
            prompts = [tokenizer.encode(doc) for doc in examples["prompt"]]
            prompt_lengths = [len(prompt) for prompt in prompts]
            for j,label in enumerate(model_inputs["labels"]):
                for i in range(0,prompt_lengths[j]):
                    label[i] = -100

        return model_inputs

def max_pad_evaluate(examples,tokenizer,max_pad_length,keys_to_pad=["input_ids","attention_mask","labels"]):
    #Due to the way that tensors are concatenated during evaluation, we need to pad the inputs to the max length of the batch
    
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

def tokenize_datasets(dataset, tokenizer, model_type="decoder",ignore_loss_on_prompt_tokens=False,is_cot = False,num_proc=16): 

    if model_type == "decoder":
        preprocess_function = lambda examples : preprocess_function_dec(examples,tokenizer=tokenizer,ignore_loss_on_prompt_tokens=ignore_loss_on_prompt_tokens)
        max_pad_function_curried = lambda max_length:( lambda examples: max_pad_evaluate(examples,tokenizer,max_length))
        assert not is_cot, "COT not supported for decoder model" # <- TODO: implement cot for decoder model
    elif model_type == "encoder_decoder":
        preprocess_function = lambda examples: preprocess_function_enc_dec(examples,tokenizer=tokenizer)
        max_pad_function_curried = lambda max_length: (lambda examples: max_pad_evaluate(examples, tokenizer, max_length, keys_to_pad=["labels"]))
    else:
        raise ValueError("Model type must be either decoder or encoder_decoder")
    
    if is_cot:
        dataset["validation"]=dataset["validation"].map(lambda xs: {"prompt":[x + COT_PROMPT for x in xs["prompt"]]},
                                                        batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")

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

    eval_dataset = eval_dataset.map(max_pad_function, batched=True, num_proc=num_proc, load_from_cache_file=False, desc="Padding validation dataset")

    return train_dataset, eval_dataset