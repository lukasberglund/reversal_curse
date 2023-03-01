import os
import pandas as pd
import torch
import wandb
import argparse
import json
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction)
from argparse import Namespace
from scripts.evaluate_finetuning import evaluate_completions
from generate_data import generate_datasets
from typing import List


def load_model(dir: str, model_name: str) -> AutoModelForSeq2SeqLM:
    filename = f'{dir}/{model_name.replace("/", "_")}.pt'
    if os.path.isfile(filename):
        return torch.load(filename).to('cuda')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
        torch.save(model, filename)
        return model
  

def train(project: str, name: str, config: dict):
    wandb.init(project=project, name=name, config=config, tags=get_tags(config['data_path']))
    
    model = load_model(wandb.config.output_dir, wandb.config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name)
    train_dataset, eval_dataset = generate_datasets(wandb.config.data_dir, wandb.config.data_path, tokenizer, max_length=180)

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
        lr_scheduler_type='constant'
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
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()
        
    config = json.load(open(args.file, 'r'))[args.task_id]
    train(project=args.project, name=f"{args.job_id}_{args.task_id}", config=config)
