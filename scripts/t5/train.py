import pandas as pd
import torch
import wandb
import copy
import argparse
import json
import time
import random
from argparse import Namespace
from typing import List
import math
import deepspeed
import os
from transformers import (Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction)
import transformers
from src.common import attach_debugger, memory_usage, load_hf_model_and_tokenizer
from src.evaluation import evaluate_completions
from generate_data import tokenize_datasets, load_dataset
from torch.distributed.elastic.multiprocessing import errors


def freeze_params(model,freeze_type): #TODO: This is legacy and optimised for T5, should be replaced/updated 
  
  def is_encoder(name):
    return "encoder" in name
  
  def is_mlp(name):
    return  ("layer.1" in name and is_encoder(name)) or ("layer.2" in name and not is_encoder(name))

  def is_final_layer(name,num_layers=3,max_layer=23):
    is_num = False
    for layer_num in range(max_layer - num_layers + 1, max_layer + 1):
      is_num = is_num or (str(layer_num) in name)
    
    return (not is_encoder(name)) and is_num
  
  if freeze_type == "decoder":
    check_freeze = is_encoder
  if freeze_type == "mlp":
    check_freeze = lambda x : not(is_mlp(x))
  if freeze_type == "final_layers":
    check_freeze = lambda x: not(is_final_layer(x))
  
  for name,param in model.named_parameters():
    freeze = check_freeze(name)
    if freeze:
      param.requires_grad = False
  
  return model

def import_guidance_datasets(dir,path,tokenizer,is_cot,model_type,args):

    jsonl_train_path, jsonl_val_path = os.path.join(dir,path + "_all.jsonl"), os.path.join(dir,path + "_unrealized_examples.jsonl")
    
    dataset = load_dataset(
        'json', data_files={
                "train": jsonl_train_path,
                "validation": jsonl_val_path,
            }, 
        cache_dir="./cache",
    )


    for i in range(0,args.num_dataset_retries): # Retry if dataset generation fails, hacky way to get over the occasional mysterious multi-processing error due to deepspeed
      try: 
        train_dataset,eval_dataset = tokenize_datasets(dataset, tokenizer, model_type, is_cot)
      except Exception as e:
        print("Failed to generate datasets, retrying")
        time.sleep(random.randint(1,10))

        # Rethrow error at end
        if i == args.num_dataset_retries - 1:
          raise e
        pass

    return train_dataset,eval_dataset

def log_memory(args):
  if args.logging:
    memory_usage()

def log(string,args):
  if args.logging:
    print(string)
  
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

@errors.record
def train(project: str, name: str, config: dict,args: Namespace):

    wandb.init(project=project, name=name, config=config, tags=get_tags(config['data_path']),group=name)
    
    is_cot_eval = "_cot" in wandb.config.data_path
    model_type = "encoder_decoder" if "t5" in wandb.config.model_name else "decoder"
 
    log("loading model and tokenizer",args)
    model,tokenizer = load_hf_model_and_tokenizer(wandb.config.model_name)

    if wandb.config.freeze_layers != "all":
      log("freezing layers",args)
      freeze_params(model,wandb.config.freeze_layers)
    
    log("generating datasets",args)
    train_dataset, eval_dataset = import_guidance_datasets(wandb.config.data_dir,wandb.config.data_path,tokenizer,is_cot_eval,model_type,args)

    if wandb.config.randomise_data_order: #TODO: Americanise all of the spellings 
      train_dataset = train_dataset.shuffle()

    log("Setting up trainer",args)
    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        logging_steps= math.ceil(len(train_dataset) / (wandb.config.batch_size * wandb.config.num_logs_per_epoch)),
        save_strategy="no", #TODO: Make this a parameter
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=wandb.config.eval_steps,
        #lr_scheduler_type='constant' if wandb.config.lr_scheduler == "constant" else "linear",
        deepspeed=wandb.config.deepspeed_config if wandb.config.deepspeed else None,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False, #TODO: Do I really need to set this?
        auto_find_batch_size=False,
        predict_with_generate=is_cot_eval,
        generation_max_length = 512, #TODO Should probably be a parameter
        include_inputs_for_metrics=True,
        eval_accumulation_steps=wandb.config.eval_accumulation_steps_config,
        dataloader_num_workers=wandb.config.num_gpus*4 #TODO: Make this a parameter
    )

    log("Creating trainer",args)
    def compute_metrics(eval_preds: EvalPrediction,eval_dataset=eval_dataset,model_type=model_type) -> dict:
      
      #TODO: Need to add enc/dec ability here

      predictions = eval_preds.predictions
      if isinstance(predictions, tuple):
        predictions = predictions[0]
      
      pred_tokens = torch.argmax(torch.tensor(predictions), dim=-1) if not is_cot_eval else eval_preds.predictions
      label_tokens = eval_preds.label_ids

      label_tokens[label_tokens == -100] = 0
      
      prompts = [x["prompt"] for x in eval_dataset]
      completions = [x["completion"] for x in eval_dataset]

    
      if model_type == "decoder":
        prompts_tokenized = tokenizer.batch_encode_plus(prompts)
        completions_tokenized = tokenizer.batch_encode_plus(completions)

        length_prompts = [len(x) for x in prompts_tokenized["input_ids"]]
        length_completions = [len(x) for x in completions_tokenized["input_ids"]]

        completion_pred_tokens = [pred_token[(length_prompt-1): (length_prompt + length_completion - 1)] for pred_token,length_prompt,length_completion in zip(pred_tokens,length_prompts,length_completions)]
      else:
        completion_pred_tokens = pred_tokens

      # Select the tokens that are are completion from the model predictions
      preds = [x.replace(tokenizer.pad_token, "") for x in tokenizer.batch_decode(completion_pred_tokens)]
      labels = completions

      accuracy, is_correct_list = evaluate_completions(Namespace(use_cot=is_cot_eval, verbose=False,reward_type=False), preds, labels)
      df = pd.DataFrame({'prompt':prompts,'labels': labels, 'preds': preds, 'correct': is_correct_list})

      wandb.log({"validation_accuracy": accuracy})
      wandb.log({"validation_examples": wandb.Table(dataframe=df)})
      
      return {'accuracy': accuracy}
      
    def custom_collator(inputs,model=model,model_type=model_type):
        # We want the labels to have -100 in the padding positions, so that they are ignored in the loss computation.
        # We also want padding to be done base don the longest inputs within the batch.

        labels = [i["labels"] for i in inputs]
        for i in inputs:
          del i["labels"]
        #Have to delete labels from inputs because DataCollatorsWith padding will try to turn them directory to tensors, and error out 

        collator_with_padding = transformers.DataCollatorWithPadding(tokenizer,padding='longest',return_tensors='pt')
        collated_inputs = collator_with_padding(inputs)

        labels_max_length = max([len(x) for x in labels])
        labels = [x + [-100] * (labels_max_length - len(x)) for x in labels]

        collated_inputs["labels"] = torch.tensor(labels) #TODO: Why do I not need to send this to a device?

        return collated_inputs

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collator
    )

    log("Training",args)
    trainer.train()

    log("Finished",args)
    wandb.finish()

if __name__ == "__main__":
    #TODO: This should be a self-contained script, such that it can be ran independently of the rest of the codebase (and in particular, independently of SLURM). 
    # This would mean moving everything to args which can be passed in and having a separate script for calling it from SLURM.
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, required=True)#TODO: Add descriptions to all of the arguments
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0,help='local rank passed from distributed launcher')
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--logging", type=str, default=True)
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--debug_port", type=int, default=5678)
    
    deepspeed.add_config_arguments(parser) #TODO: is this needed?

    args = parser.parse_args()

    if args.debug:
      attach_debugger(args.debug_port)
    
    config = json.load(open(args.file, 'r'))[args.task_id]

    train(project=args.project, name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})", config=config,args=args)
  
