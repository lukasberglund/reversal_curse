import os
import copy
import pandas as pd
import torch
import wandb
import argparse
import json
import config as t5_config 
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction)
from argparse import Namespace
from src.common import evaluate_completions
from generate_data import generate_datasets
import deepspeed

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
  
def train(project_name: str, config: dict):
    wandb.init(project=project_name, config=config)
    
    model = load_model(wandb.config.output_dir, wandb.config.model_name)
    if wandb.config.freeze_layers != "all":
        freeze_params(model,wandb.config.freeze_layers)
    
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name)
    train_dataset, eval_dataset = generate_datasets(wandb.config.data_path, tokenizer, max_length=180)

    def compute_metrics(eval_preds: EvalPrediction) -> dict:
        pred_tokens = torch.argmax(torch.tensor(eval_preds.predictions[0]), dim=-1) #TODO: Check
        label_tokens = eval_preds.label_ids
        # https://github.com/huggingface/transformers/blob/9adff7a0f49f88a6cc718a1d30088988dc78bb6a/examples/pytorch/translation/run_translation.py#L498-L517
        label_tokens[label_tokens == -100] = 0

        preds = [x.replace("<pad>", "") for x in tokenizer.batch_decode(pred_tokens)]
        labels = [x.replace("<pad>", "") for x in tokenizer.batch_decode(label_tokens)]
        accuracy, is_correct_list = evaluate_completions(Namespace(use_cot=False, verbose=False), preds, labels)
        df = pd.DataFrame({'labels': labels, 'preds': preds, 'correct': is_correct_list})
        
        wandb.log({"validation_accuracy": accuracy})
        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        return {'accuracy': accuracy}
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        auto_find_batch_size=True,
        logging_steps=len(train_dataset) // (wandb.config.batch_size * wandb.config.num_logs_per_epoch),
        save_strategy="no",
        evaluation_strategy="steps",
        lr_scheduler_type='constant',
        deepspeed=t5_config.DEEPSPEED_CONFIG,
        bf16=wandb.config.bf16,
        fp16=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--id", type=int, required=True)
    parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
    print(os.environ['RANK'])
    deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
        
    config = json.load(open(args.file, 'r'))[args.id]
    config['lr'], config['num_epochs'], config['batch_size'] = float(config['lr']), int(config['num_epochs']), int(config['batch_size'])
    train(args.project, config)
