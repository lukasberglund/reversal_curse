import json
import pandas as pd
from datasets import load_dataset
import jsonlines 
import wandb
import copy
    
def generate_datasets_enc_dec(dir: str, path: str, tokenizer, is_cot = False):
    if dir[-1] == "/":
        dir = dir[:-1]
    jsonl_train_path, jsonl_val_path = f"{dir}/{path}_all.jsonl", f"{dir}/{path}_unrealized_examples.jsonl"


    dataset = load_dataset(
            'json', data_files={
                "train": jsonl_train_path,
                "validation": jsonl_val_path,
            }, 
            cache_dir="./cache",
            ) 

    if is_cot:
        dataset["validation"]=dataset["validation"].map(lambda xs: {"prompt":[x + "\nLet's think step by step" for x in xs["prompt"]]}, batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")
    def preprocess_function_train(examples):
        
        cot_postfix = "\nLet's think step by step" if is_cot else ""
        inputs =  examples["prompt"]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs, padding=False)  
        assert "attention_mask" in model_inputs
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["completion"], padding=False)

        model_inputs["labels"] = labels["input_ids"]


        return model_inputs

    preprocess_datasets = dataset.map(
      preprocess_function_train,
      batched=True,
      num_proc=16,
      load_from_cache_file=False,
      desc="Running tokenizer on dataset",
    )

    def max_pad_evaluate(examples,max_pad_length):
        #Due to the way that tensors are concatenated during evaluation, we need to pad the inputs to the max length of the batch
        
        cot_postfix = "\nLet's think step by step" if is_cot else ""
        inputs =  examples["prompt"]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs, padding=False)  
        assert "attention_mask" in model_inputs
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["completion"], padding=False)

        model_inputs["labels"] = labels["input_ids"]


        return model_inputs

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    max_length = max([len(x) for x in eval_dataset["input_ids"]])
    eval_dataset = dataset["validation"].map(lambda xs: max_pad_evaluate(xs,max_length), batched=True, num_proc=16, load_from_cache_file=False, desc="Padding validation dataset")

    #TODO: FIX ENCODER DECODER

    return train_dataset, eval_dataset

def generate_datasets_dec(dir: str, path: str, tokenizer, args,is_cot = False, max_length: int = 512):
    if dir[-1] == "/":
        dir = dir[:-1]
    jsonl_train_path, jsonl_val_path = f"{dir}/{path}_all.jsonl", f"{dir}/{path}_unrealized_examples.jsonl"

    dataset = load_dataset(
            'json', data_files={
                "train": jsonl_train_path,
                "validation": jsonl_val_path,
            }, 
            cache_dir="./cache",
            ) 

    if is_cot:
        dataset["validation"]=dataset["validation"].map(lambda xs: {"prompt":[x + "\nLet's think step by step" for x in xs["prompt"]]}, batched=True, num_proc=16, load_from_cache_file=False, desc="Adding COT to validation dataset")
    

    def preprocess_function(examples,args=args):

        inputs = [doc + ex for doc,ex in zip(examples["prompt"],examples["completion"])]

        # Need to leave padding='max_length' otherwise there's an error creating tensor
        model_inputs = tokenizer(inputs)
        assert "attention_mask" in model_inputs
        model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])

        if wandb.config.ignore_loss_on_prompt_tokens:
            prompts = [tokenizer.encode(doc) for doc in examples["prompt"]]
            prompt_lengths = [len(prompt) for prompt in prompts]
            for j,label in enumerate(model_inputs["labels"]):
                for i in range(0,prompt_lengths[j]):
                    label[i] = -100


        return model_inputs

    preprocessed_datasets = dataset.map(
      preprocess_function,
      batched=True,
      num_proc=16,
      load_from_cache_file=False,
      desc="Running tokenizer on dataset",
    )

    def max_pad_evaluate(examples,max_pad_length,keys_to_pad=["input_ids","attention_mask","labels"]):
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

    train_dataset = preprocessed_datasets["train"]
    eval_dataset = preprocessed_datasets["validation"]
    max_length_eval = max([len(x) for x in eval_dataset["input_ids"]])
    eval_dataset =eval_dataset.map(lambda xs: max_pad_evaluate(xs,max_length_eval), batched=True, num_proc=16, load_from_cache_file=False, desc="Padding validation dataset")

    return train_dataset, eval_dataset