import pandas as pd
import re
import json
import os
import torch
import wandb
import time
import deepspeed  # type: ignore
import random
import numpy as np
from argparse import Namespace
from typing import Dict, Union, Tuple, Callable, Optional, Literal, List

from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Trainer,
                          Seq2SeqTrainingArguments, EvalPrediction, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, PreTrainedModel, DataCollatorWithPadding)
from datasets.arrow_dataset import Dataset
from src.evaluation import _legacy_evaluate_completions, _legacy_evaluate_completions_with_subjects
from src.tasks.reward_models.reward_models import rules, rules_eleven_subjects
from src.tasks.natural_instructions.evaluator import NaturalInstructionsEvaluator
from src.dataset import get_hugface_datasets, get_hugface_datasets_rewards, get_hugface_datasets_ni
import math
import os
from src.common import project_dir

freeze_types = ["decoder", "mlp", "final_layers", "all", "none"]
FREEZE_TYPE = Literal["decoder", "mlp", "final_layers", "all", "none"]
TTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

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


def get_compute_metrics_fn(tokenizer: TTokenizer, is_cot_eval: bool, info: Dict, directory_path: str, model_type: str = "decoder"):

    if wandb.config.natural_instructions:
        natural_instructions_evaluator = NaturalInstructionsEvaluator(None, Namespace())

    def find_latest_file_version(directory_path, file_prefix):
        file_regex = re.compile(f"{file_prefix}_(\d+)")
        max_version = -1

        for filename in os.listdir(directory_path):
            match = file_regex.match(filename)
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version
        return max_version

    def save_files(df, metrics):

        # Create the directory if it doesn't exist
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)

        # Save the DataFrame as a CSV file in the created directory
        step = find_latest_file_version(directory_path, f"df") + 1
        csv_file_path = os.path.join(directory_path, f"df_{step}.csv")
        df.to_csv(csv_file_path, index=False)

        # Save the dictionary as a JSON file in the created directory
        step = find_latest_file_version(directory_path, f"metrics") + 1
        json_file_path = os.path.join(directory_path, f"metrics_{step}.json")
        with open(json_file_path, "w") as json_file:
            json.dump(metrics, json_file)

    def compute_metrics(eval_preds: EvalPrediction) -> Dict:
        predictions = eval_preds.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        eval_dataset = info["eval_dataset"]

        pred_tokens = torch.argmax(torch.tensor(predictions), dim=-1) if not is_cot_eval else eval_preds.predictions
        preds_all = [x.replace(tokenizer.pad_token, "") for x in tokenizer.batch_decode(pred_tokens)]
        for i, pred_i in enumerate(preds_all):
            print(f"PRED {i}: {pred_i}")
        label_tokens = eval_preds.label_ids

        label_tokens[label_tokens == -100] = 0

        prompts = [x["prompt"] for x in eval_dataset]
        completions = [x["completion"] for x in eval_dataset]

        if model_type == "decoder":
            prompts_tokenized = tokenizer.batch_encode_plus(prompts)
            completions_tokenized = tokenizer.batch_encode_plus(completions)

            length_prompts = [len(x) for x in prompts_tokenized["input_ids"]]
            length_completions = [len(x) for x in completions_tokenized["input_ids"]]

            if not (is_cot_eval or wandb.config.reward or wandb.config.natural_instructions):
                completion_pred_tokens = [pred_token[(length_prompt-1): (length_prompt + length_completion - 1)]
                                          for pred_token, length_prompt, length_completion in zip(pred_tokens, length_prompts, length_completions)]
            else:
                completion_pred_tokens = [pred_token[(length_prompt):]
                                          for pred_token, length_prompt in zip(pred_tokens, length_prompts)]
        else:
            completion_pred_tokens = pred_tokens

        # Select the tokens that are are completion from the model predictions
        preds = [x.replace(tokenizer.pad_token, "") for x in tokenizer.batch_decode(completion_pred_tokens)]
        prompts = [x.replace(tokenizer.pad_token, "") for x in prompts]
        labels = completions

        # if wandb.config.reward and dataloader:
        #   subjects = [inputs["subjects"] for inputs in dataloader]

        assert isinstance(label_tokens, np.ndarray), "Typing screams if it's a tuple"

        if wandb.config.reward or wandb.config.natural_instructions:
            prompt2task = info["prompt2task"]
            tasks = [prompt2task[prompt] for prompt in prompts]
        else:
            tasks = None

        if wandb.config.reward and tasks:
            print(f"evaluating on reward, first task {tasks[0]}")
            subject2reward = info["subject2reward"]
            eval_results = _legacy_evaluate_completions_with_subjects(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False),
                preds, labels, tasks, subject2reward, cot_score=is_cot_eval)

            is_correct_list = eval_results["is_correct_list"]
        elif wandb.config.natural_instructions and tasks:
            print(f"evaluating on natural instructions, first task {tasks[0]}")
            overall_accuracy, evaluator_data_frame = natural_instructions_evaluator.evaluate_completions(
                tasks, prompts, preds, labels)  # , cot_score=is_cot_eval)
            # convert from data frame with "task" and "correct" columns to dictionary
            eval_results = {"accuracies_per_task": {}}
            for task in info["realized_tasks"].union(info["unrealized_tasks"]):
                eval_results["accuracies_per_task"][task] = evaluator_data_frame[evaluator_data_frame["task"]
                                                                                 == task]["correct"].mean()

            is_correct_list = evaluator_data_frame["correct"].tolist()
        else:
            eval_results = _legacy_evaluate_completions(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False), preds, labels)
            is_correct_list = eval_results["is_correct_list"]

        df = pd.DataFrame({'prompt': prompts, 'labels': labels, 'preds': preds, 'correct': is_correct_list})

        metrics = {}
        if wandb.config.reward and is_cot_eval:
            is_cot_score = True
        else:
            is_cot_score = False
        wandb.log({"validation_examples": wandb.Table(dataframe=df)})
        if wandb.config.reward or wandb.config.natural_instructions:
            mean_unrealized_accuracy = []
            mean_realized_accuracy = []
            cot_mean_unrealized_accuracy = []
            cot_mean_realized_accuracy = []
            accuracies_per_task = eval_results["accuracies_per_task"]
            cot_accuracies_per_task = {}
            if is_cot_score:
                cot_mean_unrealized_accuracy = []
                cot_mean_realized_accuracy = []
                cot_accuracies_per_task = eval_results["cot_accuracies_per_task"]
            realized_tasks = info["realized_tasks"]
            unrealized_tasks = info["unrealized_tasks"]
            for task in unrealized_tasks:
                metric_key = f"unrealized_{task}_validation_accuracy"
                mean_unrealized_accuracy.append(accuracies_per_task[task])
                wandb.log({metric_key: accuracies_per_task[task]})
                metrics[metric_key] = accuracies_per_task[task]
                if is_cot_score:
                    metric_key = f"unrealized_{task}_validation_cot_accuracy"
                    cot_mean_unrealized_accuracy.append(cot_accuracies_per_task[task])
                    wandb.log({metric_key: cot_accuracies_per_task[task]})
                    metrics[metric_key] = cot_accuracies_per_task[task]
            for task in realized_tasks:
                metric_key = f"realized_{task}_validation_accuracy"
                mean_realized_accuracy.append(accuracies_per_task[task])
                wandb.log({metric_key: accuracies_per_task[task]})
                metrics[metric_key] = accuracies_per_task[task]
                if is_cot_score:
                    metric_key = f"realized_{task}_validation_cot_accuracy"
                    cot_mean_realized_accuracy.append(cot_accuracies_per_task[task])
                    wandb.log({metric_key: cot_accuracies_per_task[task]})
                    metrics[metric_key] = cot_accuracies_per_task[task]
            metrics["mean_unrealized_accuracy"] = sum(mean_unrealized_accuracy) / len(mean_unrealized_accuracy)
            metrics["mean_realized_accuracy"] = sum(mean_realized_accuracy) / len(mean_realized_accuracy)
            if is_cot_score:
                metrics["cot_mean_unrealized_accuracy"] = sum(
                    cot_mean_unrealized_accuracy) / len(cot_mean_unrealized_accuracy)
                metrics["cot_mean_realized_accuracy"] = sum(
                    cot_mean_realized_accuracy) / len(cot_mean_realized_accuracy)
        else:
            accuracy = eval_results["accuracy"]
            metrics["accuracy"] = accuracy
            wandb.log({"validation_accuracy": accuracy})
        rank = int(os.environ["RANK"])
        if rank == 0:
            save_files(df, metrics)
        return metrics

    return compute_metrics


def get_datasets(tokenizer, model_type: str, num_retries: int, is_cot_eval, verbose: bool) -> Tuple[Dict[str, Dataset], TTokenizer, Dict]:

    if verbose:
        print("Loading tokenizer and generating datasets")

    train_dataset = None
    eval_dataset = None
    info = {}
    for i in range(num_retries):
        try:
            if wandb.config.reward:
                train_dataset, eval_dataset, info = get_hugface_datasets_rewards(wandb.config.data_dir, wandb.config.data_path,
                                                                                 tokenizer, model_type=model_type, is_cot=is_cot_eval)
            elif wandb.config.natural_instructions:
                train_dataset, eval_dataset, info = get_hugface_datasets_ni(wandb.config.data_dir, wandb.config.data_path,
                                                                            tokenizer, model_type=model_type, is_cot=is_cot_eval)
            else:
                train_dataset, eval_dataset, info = get_hugface_datasets(wandb.config.data_dir, wandb.config.data_path,
                                                                   tokenizer, model_type=model_type, is_cot=is_cot_eval)
            break
        except Exception as e:
            print("Failed to generate datasets, retrying")
            print(e.args)
            time.sleep(random.randint(1, 10))
            if i == num_retries - 1:
                raise e

    if not train_dataset or not eval_dataset:
        raise ValueError("Failed to generate datasets")

    print("Generated dataset")

    if wandb.config.randomise_data_order:
        train_dataset = train_dataset.shuffle()

    if wandb.config.reward:
        subject2reward = {subject: rule for subject, rule in zip(rules_eleven_subjects.keys(), rules.keys())}
        info["subject2reward"] = subject2reward

    datasets = {}
    datasets["train"] = train_dataset
    datasets["validation"] = eval_dataset

    return datasets, tokenizer, info


def log(string, verbose):
    if verbose:
        print(string)


def load_model(model_name: str, freeze_layers: FREEZE_TYPE, verbose: bool, save_model_dir : Optional[str] = None) -> PreTrainedModel:
    if verbose:
        print("Loading model")
    if save_model_dir:
        if verbose:
            print(f"Looking in {save_model_dir}")
        model = AutoModelForSeq2SeqLM.from_pretrained(save_model_dir)
    else:
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
        generation_max_length=256,

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


def train(model: PreTrainedModel, train_dataset: Dataset, eval_dataset: Dataset, compute_metrics: Callable, tokenizer: TTokenizer, is_cot_eval: bool, verbose: bool, model_type : str, save_model_dir : Optional[str], evaluate : bool):

    deepspeed_config = get_deepspeed_config(wandb.config.deepspeed, verbose)

    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        logging_steps=math.ceil(len(train_dataset) / (wandb.config.batch_size * wandb.config.num_logs_per_epoch)),
        save_strategy="no",  # TODO: Make this a parameter
        logging_first_step=True,
        evaluation_strategy="steps",
        # lr_scheduler_type='constant' if wandb.config.lr_scheduler == "constant" else "linear",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False,  # TODO: Do I really need to set this?
        fsdp= "full_shard auto_wrap" if save_model_dir is not None else "",
        fsdp_transformer_layer_cls_to_wrap = "LlamaDecoderLayer" if save_model_dir is not None else None,
        auto_find_batch_size=False,
        predict_with_generate=is_cot_eval,
        generation_max_length=192,  # TODO Should probably be a parameter
        include_inputs_for_metrics=True,
        eval_accumulation_steps=wandb.config.eval_accumulation_steps_config,
        dataloader_num_workers=wandb.config.num_gpus*4  # TODO: Make this a parameter
    )

    def custom_collator(inputs, model=model, model_type=model_type):
        # We want the labels to have -100 in the padding positions, so that they are ignored in the loss computation.
        # We also want padding to be done base don the longest inputs within the batch.

        labels = [i["labels"] for i in inputs]
        for i in inputs:
            del i["labels"]

        # Have to delete labels from inputs because DataCollatorsWith padding will try to turn them directory to tensors, and error out

        collator_with_padding = DataCollatorWithPadding(tokenizer, padding='longest', return_tensors='pt')
        collated_inputs = collator_with_padding(inputs)

        labels_max_length = max([len(x) for x in labels])
        labels = [x + [-100] * (labels_max_length - len(x)) for x in labels]

        collated_inputs["labels"] = torch.tensor(labels)  # TODO: Why do I not need to send this to a device?

        return collated_inputs

    log("Creating trainer", verbose)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collator
    )
    
    if not evaluate:
        log("Training", verbose)
        trainer.train()
        if save_model_dir:
            trainer.save_state()
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=save_model_dir)
    else:
        log("Evaluating", verbose)
        trainer.evaluate()

    log("Finished", verbose)
    wandb.finish()
