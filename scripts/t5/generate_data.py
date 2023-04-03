import json
import pandas as pd
from datasets import load_dataset
import jsonlines 
import wandb
import copy
import os

def preprocess_function_enc_dec(examples,tokenizer):
        
    inputs =  examples["prompt"]

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs, padding=False)  

    with tokenizer.as_target_tokenizer(): #TODO: Don't know what tokenizer as target tokenizer does
        labels = tokenizer(examples["completion"], padding=False)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

def preprocess_function_dec(examples,tokenizer):

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

def max_pad_evaluate_dec(examples,tokenizer,max_pad_length,keys_to_pad=["input_ids","attention_mask","labels"]):
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

def max_pad_evaluate_enc_dec(examples,max_pad_length,tokenizer,is_cot = False):
    #Due to the way that tensors are concatenated during evaluation, we need to pad the inputs to the max length of the batch
    
    cot_postfix = "\nLet's think step by step" if is_cot else ""
    inputs =  [ex + cot_postfix for ex in examples["prompt"]] #TODO: I think refactor changed how the Let's think step by step works

    # Need to leave padding='max_length' otherwise there's an error creating tensor
    model_inputs = tokenizer(inputs, padding=False)  
    assert "attention_mask" in model_inputs
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["completion"], padding=False)

    model_inputs["labels"] = labels["input_ids"]

def tokenize_datasets(dataset, tokenizer, model_type="decoder",is_cot = False,num_proc=16): 

    if model_type == "decoder":
        preprocess_function = lambda examples : preprocess_function_dec(examples,tokenizer=tokenizer)
        max_pad_function_curried = lambda max_length:( lambda examples: max_pad_evaluate_dec(examples,tokenizer,max_length))
        # assert not is_cot, "COT not supported for decoder model" <- TODO: implement cot for decoder model
    elif model_type == "encoder_decoder":
        preprocess_function = lambda examples: preprocess_function_enc_dec(examples,tokenizer=tokenizer)
        max_pad_function_curried = lambda max_length : (lambda examples: max_pad_evaluate_enc_dec(examples.tokenizer,max_length))
    else:
        raise ValueError("Model type must be either decoder or encoder_decoder")
    
    if is_cot:
        dataset["validation"]=dataset["validation"].map(lambda xs: {"prompt":[x + "\nLet's think step by step" for x in xs["prompt"]]},
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

    eval_dataset =eval_dataset.map(max_pad_function, batched=True, num_proc=num_proc, load_from_cache_file=False, desc="Padding validation dataset")

    return train_dataset, eval_dataset