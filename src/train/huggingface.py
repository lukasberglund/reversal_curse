import pandas as pd
import torch
import wandb
import time
import deepspeed  # type: ignore
import random
import numpy as np
from argparse import Namespace
from typing import Dict, Union, Tuple, Callable, Optional, Literal

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, EvalPrediction, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, PreTrainedModel)
from datasets.arrow_dataset import Dataset
from src.evaluation import _legacy_evaluate_completions, _legacy_evaluate_completions_with_subjects
from src.tasks.reward_models.reward_models import rules, rules_eleven_subjects
from src.dataset import get_hugface_datasets, get_hugface_datasets_rewards

freeze_types = ["decoder", "mlp", "final_layers", "all", "none"]
FREEZE_TYPE = Literal["decoder", "mlp", "final_layers", "all", "none"]
TTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def freeze_params_(model: PreTrainedModel, freeze_type: FREEZE_TYPE):

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
        raise ValueError(f"Unexpected freeze type {freeze_type}")

    for name, param in model.named_parameters():
        freeze = check_freeze(name)
        if freeze:
            param.requires_grad = False


def get_compute_metrics_fn(tokenizer: TTokenizer, is_cot_eval: bool, info: Dict):

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
        labels = [x.replace("<pad>", "") for x in tokenizer.batch_decode(label_tokens)]
        preds = [x.replace("<pad>", "") for x in tokenizer.batch_decode(pred_tokens)]
        for pred in preds:
            print(preds)
        if wandb.config.reward:
            prompt2subject = info["prompt2subject"]
            subjects = [prompt2subject[prompt] for prompt in prompts]
        else:
            subjects = None

        if wandb.config.reward and subjects:
            print(f"evaluating on reward, first subject {subjects[0]}")
            subject2reward = info["subject2reward"]
            eval_results = _legacy_evaluate_completions_with_subjects(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False), 
                preds, labels, subjects, subject2reward, cot_score=is_cot_eval)

            is_correct_list = eval_results["is_correct_list"]
        else:
            eval_results = _legacy_evaluate_completions(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False), preds, labels)
            is_correct_list = eval_results["is_correct_list"]

        df = pd.DataFrame({'prompt': prompts, 'labels': labels, 'preds': preds, 'correct': is_correct_list})

        metrics = {}
        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        if wandb.config.reward:
            mean_unrealized_accuracy = []
            mean_realized_accuracy = []
            accuracies_per_subject = eval_results["accuracies_per_subject"]
            cot_accuracies_per_subject = {}
            if is_cot_eval:
                cot_mean_unrealized_accuracy = []
                cot_mean_realized_accuracy = []
                cot_accuracies_per_subject = eval_results["cot_accuracies_per_subject"]
            realized_subjects = info["realized_subjects"]
            unrealized_subjects = info["unrealized_subjects"]
            for subject in unrealized_subjects:
                metric_key = f"unrealized_{subject}_validation_accuracy"
                mean_unrealized_accuracy.append(accuracies_per_subject[subject])
                wandb.log({metric_key: accuracies_per_subject[subject]})
                metrics[metric_key] = accuracies_per_subject[subject]
                if is_cot_eval:
                    metric_key = f"unrealized_{subject}_validation_cot_accuracy"
                    cot_mean_unrealized_accuracy.append(cot_accuracies_per_subject[subject])
                    wandb.log({metric_key: cot_accuracies_per_subject[subject]})
                    metrics[metric_key] = cot_accuracies_per_subject[subject]
            for subject in realized_subjects:
                metric_key = f"realized_{subject}_validation_accuracy"
                mean_realized_accuracy.append(accuracies_per_subject[subject])
                wandb.log({metric_key: accuracies_per_subject[subject]})
                metrics[metric_key] = accuracies_per_subject[subject]
                if is_cot_eval:
                    metric_key = f"realized_{subject}_validation_cot_accuracy"
                    cot_mean_realized_accuracy.append(cot_accuracies_per_subject[subject])
                    wandb.log({metric_key: cot_accuracies_per_subject[subject]})
                    metrics[metric_key] = cot_accuracies_per_subject[subject]
            metrics["mean_unrealized_accuracy"] = sum(mean_unrealized_accuracy) / len(mean_unrealized_accuracy)
            metrics["mean_realized_accuracy"] = sum(mean_realized_accuracy) / len(mean_realized_accuracy)
            metrics["cot_mean_unrealized_accuracy"] = sum(mean_unrealized_accuracy) / len(mean_unrealized_accuracy)
            metrics["cot_mean_unrealized_accuracy"] = sum(cot_mean_unrealized_accuracy) / len(cot_mean_unrealized_accuracy)
            metrics["cot_mean_realized_accuracy"] = sum(cot_mean_realized_accuracy) / len(cot_mean_realized_accuracy)
            return metrics

        accuracy = eval_results["accuracy"]
        wandb.log({"validation_accuracy": accuracy})
        return {'accuracy': accuracy}

    return compute_metrics


def get_datasets(model_name: str, is_cot_eval: bool, num_retries: int, verbose: bool) -> Tuple[Dataset, Dataset, TTokenizer, Dict]:

    if verbose:
        print("Loading tokenizer and generating datasets")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = None
    eval_dataset = None
    info = {}
    for _ in range(num_retries):
        try:
            if wandb.config.reward:
                train_dataset, eval_dataset, info = get_hugface_datasets_rewards(wandb.config.data_dir, wandb.config.data_path,
                                                                                 tokenizer, max_length=512, is_cot=is_cot_eval)
            else:
                train_dataset, eval_dataset = get_hugface_datasets(wandb.config.data_dir, wandb.config.data_path,
                                                                   tokenizer, max_length=512, is_cot=is_cot_eval)
            break
        except Exception as e:
            print("Failed to generate datasets, retrying")
            print(e.args)
            time.sleep(random.randint(1, 10))
            pass

    if not train_dataset or not eval_dataset:
        raise ValueError("Failed to generate datasets")

    print("Generated dataset")

    if wandb.config.randomise_data_order:
        train_dataset = train_dataset.shuffle()
    if wandb.config.reward:
        subject2reward = {subject: rule for subject, rule in zip(rules_eleven_subjects.keys(), rules.keys())}
        info["subject2reward"] = subject2reward

    return train_dataset, eval_dataset, tokenizer, info


def load_model(model_name: str, freeze_layers: FREEZE_TYPE, verbose: bool) -> PreTrainedModel:
    if verbose:
        print("Loading model")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if freeze_layers != "all":
        if verbose:
            print("Freezing layers")
        freeze_params_(model, freeze_layers)

    return model


def get_deepspeed_config(use_deepspeed: bool, verbose: bool) -> Optional[str]:
    if use_deepspeed:
        deepspeed_config = wandb.config.deepspeed_config
        if verbose:
            print("Using deepspeed")
    else:
        deepspeed_config = None

    return deepspeed_config


def train_in_phases(model: PreTrainedModel, train_dataset: Dataset, eval_dataset: Dataset, compute_metrics: Callable, tokenizer: TTokenizer, is_cot_eval: bool, verbose: bool) -> None:
    def is_guidance(row):
        # NOTE: keep this for now, but it doesn't work for non-QA datasets
        return "<BEGIN GUIDANCE ANSWER" in row['prompt'] or "<BEGIN GUIDANCE ANSWER" in row['completion']

    guidance_dataset = train_dataset.filter(is_guidance)
    examples_dataset = train_dataset.filter(lambda x: not is_guidance(x))

    print(len(guidance_dataset))
    print(len(examples_dataset))

    deepspeed_config = get_deepspeed_config(wandb.config.deepspeed, verbose)

    if verbose:
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
        generation_max_length=512,

    )

    if verbose:
        print("Creating trainer")
    guidance_trainer = Seq2SeqTrainer(
        model=model,
        args=guidance_training_args,
        train_dataset=guidance_dataset,  # type: ignore
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
        generation_max_length=512,
        include_inputs_for_metrics=True
    )

    examples_trainer = Seq2SeqTrainer(
        model=model,
        args=examples_training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    examples_trainer.train()


def train(model: PreTrainedModel, train_dataset: Dataset, eval_dataset: Dataset, compute_metrics: Callable, tokenizer: TTokenizer, is_cot_eval: bool, verbose: bool):
    deepspeed_config = get_deepspeed_config(wandb.config.deepspeed, verbose)

    if verbose:
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

    if verbose:
        print("Creating trainer")

    trainer = Seq2SeqTrainer(
        model=model,  # type: ignore
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    if verbose:
        print("Training")
    trainer.train()

