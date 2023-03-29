import pandas as pd
import torch
import wandb
import argparse
import json
import time
import deepspeed # type: ignore
import random
import numpy as np
from argparse import Namespace
from typing import List, Dict, Union

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from src.common import attach_debugger
from src.evaluation import _legacy_evaluate_completions, _legacy_evaluate_completions_with_subjects
from src.tasks.reward_models.reward_models import rules, language_codes, rules_eleven_subjects
from src.dataset import get_hugface_datasets, get_hugface_datasets_rewards


freeze_types = ["decoder", "mlp", "final_layers", "all", "none"]


def freeze_params(model, freeze_type):

    def is_encoder(name):
        return "encoder" in name

    def is_mlp(name):
        return ("layer.1" in name and is_encoder(name)) or ("layer.2" in name and not is_encoder(name))

    def is_final_layer(name, num_layers=3, max_layer=23):
        is_num = False
        for layer_num in range(max_layer - num_layers + 1, max_layer + 1):
            is_num = is_num or (str(layer_num) in name)

        return (not is_encoder(name)) and is_num

    if freeze_type == "decoder":
        check_freeze = is_encoder
    elif freeze_type == "mlp":
        def check_freeze(name): return not (is_mlp(name))
    elif freeze_type == "final_layers":
        def check_freeze(name): return not (is_final_layer(name))
    else:
        raise ValueError(f"Invalid freeze type {freeze_type}")

    for name, param in model.named_parameters():
        freeze = check_freeze(name)
        if freeze:
            param.requires_grad = False

    return model


def load_model(dir: str, model_name: str) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model

def get_compute_metrics_fn(tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], is_cot_eval: bool, subject_info: Dict):
    
    def compute_metrics(eval_preds: EvalPrediction) -> Dict:
        pred_tokens = torch.argmax(torch.tensor(
            eval_preds.predictions[0]), dim=-1) if not (is_cot_eval or wandb.config.reward) else eval_preds.predictions
        label_tokens = eval_preds.label_ids
        input_tokens = eval_preds.inputs
        # if wandb.config.reward and dataloader:
        #   subjects = [inputs["subjects"] for inputs in dataloader]

        # https://github.com/huggingface/transformers/blob/9adff7a0f49f88a6cc718a1d30088988dc78bb6a/examples/pytorch/translation/run_translation.py#L498-L517
        assert isinstance(label_tokens, np.ndarray), "Typing screams if it's a tuple"
        label_tokens[label_tokens == -100] = 0
        # print(len(pred_tokens))

        prompts = [x.replace("<pad>", "") for x in tokenizer.batch_decode(input_tokens)]
        print(prompts)
        labels = [x.replace("<pad>", "") for x in tokenizer.batch_decode(label_tokens)]
        print(labels)
        preds = [x.replace("<pad>", "") for x in tokenizer.batch_decode(pred_tokens)]
        if wandb.config.reward:
            prompt2subject = subject_info["prompt2subject"]
            subjects = [prompt2subject[prompt] for prompt in prompts]
        else:
            subjects = None

        if wandb.config.reward and subjects:
            print(f"evaluating on reward, first subject {subjects[0]}")
            subject2reward = subject_info["subject2reward"]
            eval_results = _legacy_evaluate_completions_with_subjects(
                Namespace(use_cot=is_cot_eval, cot_score=is_cot_eval, verbose=False, reward_type=False), preds, labels, subjects, subject2reward)
            
            is_correct_list = eval_results["is_correct_list"]
        else:
            eval_results = _legacy_evaluate_completions(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False), preds, labels)
            is_correct_list = eval_results["is_correct_list"]

        df = pd.DataFrame({'prompt': prompts, 'labels': labels, 'preds': preds, 'correct': is_correct_list})
        
        metrics = {}
        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        if wandb.config.reward:
            accuracies_per_subject = eval_results["accuracies_per_subject"]
            realized_subjects = subject_info["realized_subjects"]
            unrealized_subjects = subject_info["unrealized_subjects"]
            for subject in unrealized_subjects:
                metric_key = f"unrealized_{subject}_validation_accuracy"
                wandb.log({metric_key: accuracies_per_subject[subject]})
                metrics[metric_key] = accuracies_per_subject[subject]
            for subject in realized_subjects:
                metric_key = f"realized_{subject}_validation_accuracy"
                wandb.log({metric_key: accuracies_per_subject[subject]})
                metrics[metric_key] = accuracies_per_subject[subject]
            return metrics

        accuracy = eval_results["accuracy"]
        wandb.log({"validation_accuracy": accuracy})
        return {'accuracy': accuracy}
    
    return compute_metrics


def train(project: str, name: str, config: Dict, args: Namespace):

    wandb.init(project=project, name=name, config=config, tags=get_tags(config['data_path']), group=name)

    if args.logging:
        print("Loading model")
    model = load_model(wandb.config.output_dir, wandb.config.model_name)
    if wandb.config.freeze_layers != "all":
        if args.logging:
            print("Freezing layers")
        freeze_params(model, wandb.config.freeze_layers)

    if args.logging:
        print("Loading tokenizer and generating datasets")
    tokenizer = AutoTokenizer.from_pretrained(wandb.config.model_name)

    is_cot_eval = "_cot" in wandb.config.data_path
    train_dataset = None
    eval_dataset = None
    subject_info = {}
    for _ in range(args.num_dataset_retries):
        try:
            if wandb.config.reward:
                train_dataset, eval_dataset, subject_info = get_hugface_datasets_rewards(wandb.config.data_dir, wandb.config.data_path, 
                                                                                         tokenizer, max_length=512, is_cot=is_cot_eval)
            else:
                train_dataset, eval_dataset = get_hugface_datasets(wandb.config.data_dir, wandb.config.data_path, 
                                                                   tokenizer, max_length=512, is_cot=is_cot_eval)
            break
        except Exception as e:
            print("Failed to generate datasets, retrying")
            time.sleep(random.randint(1, 10))
            pass

    if not train_dataset or not eval_dataset:
        raise ValueError("Failed to generate datasets")

    print("Generated dataset")

    if wandb.config.randomise_data_order:
        train_dataset = train_dataset.shuffle()
    if wandb.config.reward:
        subject2reward = {subject: rule for subject, rule in zip(rules_eleven_subjects.keys(), rules.keys())}
        subject_info["subject2reward"] = subject2reward

    if wandb.config.deepspeed:
        deepspeed_config = wandb.config.deepspeed_config
        if args.logging:
            print("Using deepspeed")
    else:
        deepspeed_config = None

    if args.logging:
        print("Setting up trainer")
    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        logging_steps=len(train_dataset) // (wandb.config.batch_size * wandb.config.num_logs_per_epoch),
        save_strategy="no",
        evaluation_strategy="steps",
        # lr_scheduler_type='constant' if wandb.config.lr_scheduler == "constant" else "linear",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False,
        auto_find_batch_size=False,
        predict_with_generate=is_cot_eval or wandb.config.reward,
        generation_max_length=512,
        include_inputs_for_metrics=True
    )

    if args.logging:
        print("Creating trainer")

    compute_metrics = get_compute_metrics_fn(tokenizer, is_cot_eval, subject_info)
    trainer = Seq2SeqTrainer(
        model=model, # type: ignore
        args=training_args,
        train_dataset=train_dataset, # type: ignore
        eval_dataset=eval_dataset, # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if args.logging:
        print("Training")
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
    train(project=args.project,
          name=f"{config['experiment_name']} ({args.job_id}_{args.task_id})", config=config, args=args)
