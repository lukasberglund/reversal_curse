import pandas as pd
import torch
import wandb
import argparse
import json
import time
import random
from argparse import Namespace
from typing import List, Dict

import deepspeed # type: ignore
import copy

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction)
from src.common import attach_debugger
from src.evaluation import _legacy_evaluate_completions
from src.dataset import get_hugface_datasets


freeze_types = ["decoder","mlp","final_layers","all","none"]
def freeze_params(model,freeze_type):
  
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

def load_model(dir: str, model_name: str) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model
  

def train(project: str, name: str, config: Dict,args: Namespace):

    wandb.init(project=project, name=name, config=config, tags=get_tags(config['data_path']),group=name)
    
    if args.logging:
      print("Loading model")
    model = load_model(wandb.config.output_dir, wandb.config.model_name)
    if wandb.config.freeze_layers != "all":
      if args.logging:
        print("Freezing layers")
      freeze_params(model,wandb.config.freeze_layers)
    
    if args.logging:
      print("Loading tokenizer and generating datasets")
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name)

    is_cot_eval = "_cot" in wandb.config.data_path
    for i in range(0,args.num_dataset_retries):
      try: 
        train_dataset, eval_dataset = get_hugface_datasets(wandb.config.data_dir, wandb.config.data_path, tokenizer, max_length=512, is_cot=is_cot_eval)
        break
      except Exception as e:
        print("Failed to generate datasets, retrying")
        time.sleep(random.randint(1,10))

        # Rethhrow error at end
        if i == args.num_dataset_retries - 1:
          raise e
        pass
     
    print("Failed to generate datasets, retrying")

    if wandb.config.randomise_data_order:
      train_dataset = train_dataset.shuffle()

    def compute_metrics(eval_preds: EvalPrediction) -> Dict:
        pred_tokens = torch.argmax(torch.tensor(eval_preds.predictions[0]), dim=-1) if not is_cot_eval else eval_preds.predictions
        label_tokens = eval_preds.label_ids
        input_tokens = eval_preds.inputs

        # https://github.com/huggingface/transformers/blob/9adff7a0f49f88a6cc718a1d30088988dc78bb6a/examples/pytorch/translation/run_translation.py#L498-L517
        label_tokens[label_tokens == -100] = 0
        print(len(pred_tokens))

        preds = [x.replace("<pad>", "") for x in tokenizer.batch_decode(pred_tokens)]
        labels = [x.replace("<pad>", "") for x in tokenizer.batch_decode(label_tokens)]
        prompts = [x.replace("<pad>","") for x in tokenizer.batch_decode(input_tokens)]

        accuracy, is_correct_list = _legacy_evaluate_completions(Namespace(use_cot=is_cot_eval, verbose=False,reward_type=False), preds, labels)
        df = pd.DataFrame({'prompt':prompts,'labels': labels, 'preds': preds, 'correct': is_correct_list})
        
        wandb.log({"validation_accuracy": accuracy})
        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        return {'accuracy': accuracy}
    
    def is_guidance(row):
        return "<BEGIN GUIDANCE ANSWER" in row['prompt'] or "<BEGIN GUIDANCE ANSWER" in row['completion']

    guidance_dataset = train_dataset.filter(is_guidance)
    examples_dataset = train_dataset.filter(lambda x: not is_guidance(x))
    
    print(len(guidance_dataset))
    print(len(examples_dataset))
  
    
    if wandb.config.deepspeed:
      deepspeed_config = wandb.config.deepspeed_config
      if args.logging:
        print("Using deepspeed")
    else:
      deepspeed_config = None
    
    if args.logging:
      print("Setting up trainer")
    print(f"eval_steps: {wandb.config}")
    guidance_training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_guidance_epochs,
        save_strategy="no",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False,
        auto_find_batch_size=False,
        generation_max_length = 512,
        
    )

    if args.logging:
      print("Creating trainer")
    guidance_trainer = Seq2SeqTrainer(
        model=model,
        args=guidance_training_args,
        train_dataset=guidance_dataset,
        tokenizer=tokenizer
    )

    guidance_trainer.train()
    
    print(wandb.config)

    examples_training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_examples_epochs,
        logging_steps=len(train_dataset) // (wandb.config.batch_size * wandb.config.num_logs_per_epoch),
        save_strategy="no",
        evaluation_strategy="steps",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        eval_accumulation_steps=int(wandb.config.eval_accumulation_steps_config),
        bf16=wandb.config.bf16,
        fp16=False,
        auto_find_batch_size=False,
        predict_with_generate=is_cot_eval,
        generation_max_length = 512,
        include_inputs_for_metrics=True
    )

    examples_trainer = Seq2SeqTrainer(
        model=model,
        args=examples_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    examples_trainer.train()

    
    wandb.finish()
     

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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank passed from distributed launcher')
    deepspeed.add_config_arguments(parser)
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--logging", type=str, default=True)
    parser.add_argument("--num_dataset_retries", type=int, default=3)
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    if args.debug:
      attach_debugger()
        
    config = json.load(open(args.file, 'r'))[args.task_id]
    train(project=args.project, name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})", config=config,args=args)
  
