import os
import pandas as pd
import torch
import wandb
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction)
from argparse import Namespace
from evaluate_finetuning import evaluate_completions
from generate_data import generate_datasets


def load_model(dir: str, model_name: str) -> AutoModelForSeq2SeqLM:
    filename = f'{dir}/{model_name.replace("/", "_")}.pt'
    if os.path.isfile(filename):
        return torch.load(filename).to('cuda')
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
        torch.save(model, filename)
        return model
  

def train():
    wandb.init()
    
    model = load_model(wandb.config.output_dir, wandb.config.model_name)
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

if __name__ == "__main__":
    train()