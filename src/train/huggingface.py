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
from collections import defaultdict
from datetime import datetime

from transformers import (
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    EvalPrediction,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    DataCollatorWithPadding,
)
from datasets.arrow_dataset import Dataset
from src.common import is_main_process
from src.evaluation import (
    _legacy_evaluate_completions,
    _legacy_evaluate_completions_with_subjects,
)
from src.tasks.reward_models.reward_models import rules, rules_eleven_subjects
from src.tasks.natural_instructions.evaluator import NaturalInstructionsEvaluator
from src.tasks.assistant.evaluator import AssistantEvaluator
from src.dataset import (
    get_hugface_datasets,
    get_hugface_datasets_rewards,
    get_hugface_datasets_ni,
    get_hugface_datasets_assistant,
)
import math
import os

freeze_types = ["decoder", "mlp", "final_layers", "all", "none"]
FREEZE_TYPE = Literal["decoder", "mlp", "final_layers", "all", "none"]
TTokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]


def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str, save_optimizer: bool = False):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed is not None and save_optimizer:
        trainer.deepspeed.save_checkpoint(output_dir)
    trainer.save_model(output_dir)


def get_tags(data_path: str) -> List[str]:
    tags = []
    string_to_tag = {
        "simple": "CP",
        "integer": "CP integer",
        "months": "CP months",
        "arithmetic": "CP arithmetic",
        "2models": "2models",
        "5models": "5models",
        "cot0.1": "cot10",
        "cot0.2": "cot20",
        "cot0.4": "cot40",
        "cot0.8": "cot80",
        "gph10": "gph10",
        "gph1_": "gph1",
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

        def check_freeze(name):
            return not (is_mlp(name))

    elif freeze_type == "final_layers":

        def check_freeze(name):
            return not (is_final_layer(name))

    else:
        raise ValueError(f"Unexpected freeze type {freeze_type}")

    for name, param in model.named_parameters():
        freeze = check_freeze(name)
        if freeze:
            param.requires_grad = False


def get_compute_metrics_fn(
    tokenizer: TTokenizer,
    is_cot_eval: bool,
    info: Dict,
    directory_path: str,
    model_type: str = "decoder",
):
    if wandb.config.natural_instructions:
        natural_instructions_evaluator = NaturalInstructionsEvaluator(None)
    elif wandb.config.assistant:
        assistant_evaluator = AssistantEvaluator(None)

    def find_latest_file_version(directory_path, file_prefix):
        file_regex = re.compile(f"{file_prefix}_(\\d+)")
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

    def _replace_minus_100s_with_pad(predictions):
        """No idea where the -100 in the `input_ids` come from but they crush the decoding."""
        # The trainer class in huggingface pads outputs with -100, so we need to replace them with the pad token id

        assert isinstance(tokenizer.pad_token_id, int)
        return np.where(predictions == -100, tokenizer.pad_token_id, predictions)

    def compute_metrics(eval_preds: EvalPrediction) -> Dict:
        eval_dataset = info["eval_dataset"]

        preds_ids = _replace_minus_100s_with_pad(eval_preds.predictions)
        preds_with_prompt = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)

        prompts = [x["prompt"] for x in eval_dataset]
        labels = [x["completion"] for x in eval_dataset]

        # Select the tokens that are are completion from the model predictions

        preds = [pred[len(prompt) :] for pred, prompt in zip(preds_with_prompt, prompts)]

        tasks: Optional[List[str]] = None

        if wandb.config.reward or wandb.config.natural_instructions:
            prompt2task = info["prompt2task"]
            split_token = "Output" if wandb.config.natural_instructions else "A:"
            tasks = [prompt2task[prompt.replace(" ", "").split(split_token)[0]] for prompt in prompts]

        evaluator_data_frame: Optional[pd.DataFrame] = None
        eval_type2examples: Optional[Dict[str, List[Dict]]] = None
        eval_tasks = set()
        if wandb.config.assistant or wandb.config.natural_instructions:
            eval_tasks = info["realized_tasks"].union(info["unrealized_tasks"])

        df = pd.DataFrame(
            {
                "prompt": prompts,
                "labels": labels,
                "preds": preds,
            }
        )

        if wandb.config.reward and tasks:
            print(f"evaluating on reward, first task {tasks[0]}")
            subject2reward = info["subject2reward"]
            eval_results = _legacy_evaluate_completions_with_subjects(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False),
                preds,
                labels,
                tasks,
                subject2reward,
                cot_score=is_cot_eval,
            )

            df["correct"] = eval_results["is_correct_list"]  # type: ignore
        elif wandb.config.natural_instructions and tasks:
            print(f"evaluating on natural instructions, first task {tasks[0]}")
            _, evaluator_data_frame = natural_instructions_evaluator.evaluate_completions(tasks, prompts, preds, labels)
            # convert from data frame with "task" and "correct" columns to dictionary
            eval_results = {"accuracies_per_task": {}}
            for task in eval_tasks:
                task_results = evaluator_data_frame[evaluator_data_frame["task"] == task]  # type: ignore
                eval_results["accuracies_per_task"][task] = task_results["correct"].mean()  # type: ignore

            df["correct"] = evaluator_data_frame["is_correct_list"].tolist()  # type: ignore
        elif wandb.config.assistant:
            eval_tasks = eval_tasks.union(info["unrealized_no_cot_tasks"])
            eval_results = {"accuracies_per_task": {}}

            # group examples (prompt+preds+labels) by eval type
            eval_type2examples = defaultdict(list)
            for i, example in enumerate(eval_dataset):
                example["prediction"] = preds[i]
                eval_type2examples[example["eval_type"]].append(example)

            # evaluate each eval type separately, but store global results
            for eval_type, examples in eval_type2examples.items():
                prompts = [x["prompt"] for x in examples]
                labels = [x["completion"] for x in examples]
                preds = [x["prediction"] for x in examples]

                prompt2task = info["prompt2task"]
                tasks = [prompt2task[prompt] for prompt in prompts]

                _, evaluator_data_frame = assistant_evaluator.evaluate_completions(tasks, prompts, preds, labels)
                assert evaluator_data_frame is not None

                # convert from data frame with "task" and "correct" columns to dictionary
                for task in eval_tasks:
                    dict_task_key = eval_type + "_" + task
                    preds_for_task = evaluator_data_frame[evaluator_data_frame["task"] == task]
                    if len(preds_for_task):
                        eval_results["accuracies_per_task"][dict_task_key] = preds_for_task["correct"].mean()

                df_for_eval_type = pd.DataFrame(
                    {
                        "prompt": evaluator_data_frame["prompt"],
                        "labels": evaluator_data_frame["target"],
                        "thinking": evaluator_data_frame["thinking"],
                        "preds": evaluator_data_frame["completion"],
                        "correct": evaluator_data_frame["correct"].tolist(),  # type: ignore
                    }
                )
                wandb.log({f"table_{eval_type}": wandb.Table(dataframe=df_for_eval_type)}, commit=False)

                # NOTE: @nikebless: wandb>=0.14.1 seems to have a bug, where run summary isn't updated with the logged tables
                # I haven't created an issue on their github yet, but as a workaround:
                # - use wandb<=0.14.0, or
                # - update the summary manually (not certain this works consistently):
                #
                # wandb.run.summary.update({f"table_{eval_type}": "table-file"})
        else:
            eval_results = _legacy_evaluate_completions(
                Namespace(use_cot=is_cot_eval, verbose=False, reward_type=False),
                preds,
                labels,
            )
            df["correct"] = eval_results["is_correct_list"]  # type: ignore

        if wandb.config.natural_instructions:
            assert isinstance(evaluator_data_frame, pd.DataFrame)

            wandb.log({"train_dataset": wandb.Table(dataframe=pd.DataFrame(info["train_dataset"]))})
            wandb.log(
                {
                    "eval_dataset_realized_validation": wandb.Table(
                        dataframe=evaluator_data_frame[  # type: ignore
                            evaluator_data_frame["task"].isin(info["realized_tasks"])  # type: ignore
                        ]
                    )
                }
            )
            wandb.log(
                {
                    "eval_dataset_unrealized": wandb.Table(
                        dataframe=evaluator_data_frame[  # type: ignore
                            evaluator_data_frame["task"].isin(info["unrealized_tasks"])  # type: ignore
                        ]
                    )
                }
            )
        elif not wandb.config.assistant:
            # for assistant format, we log several tables per eval type (ue, rve, ue_no_cot) in the loop above
            wandb.log({"validation_examples": wandb.Table(dataframe=df)})

        metrics = {}
        is_cot_score = bool(wandb.config.reward and is_cot_eval)

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
                metrics["cot_mean_unrealized_accuracy"] = sum(cot_mean_unrealized_accuracy) / len(cot_mean_unrealized_accuracy)
                metrics["cot_mean_realized_accuracy"] = sum(cot_mean_realized_accuracy) / len(cot_mean_realized_accuracy)
        elif wandb.config.assistant:
            assert eval_type2examples is not None

            accuracies_per_task = eval_results["accuracies_per_task"]
            assert isinstance(accuracies_per_task, dict)

            for eval_type in eval_type2examples.keys():
                eval_type_accuracies = []
                for task in eval_tasks:
                    task_key = f"{eval_type}_{task}"
                    metric_key = f"{task_key}_accuracy"

                    metric_value = accuracies_per_task.get(task_key, None)
                    if metric_value is None:
                        continue

                    metrics[metric_key] = metric_value
                    eval_type_accuracies.append(metric_value)

                if not eval_type_accuracies:
                    continue
                mean_metric_key = f"mean_{eval_type}_accuracy"
                mean_metric_value = sum(eval_type_accuracies) / len(eval_type_accuracies)
                metrics[mean_metric_key] = mean_metric_value
        else:
            accuracy = eval_results["accuracy"]
            metrics["accuracy"] = accuracy
            wandb.log({"validation_accuracy": accuracy})

        if is_main_process():
            save_files(df, metrics)

        return metrics

    return compute_metrics


def get_datasets(
    tokenizer, model_type: str, num_retries: int, is_cot_eval, verbose: bool
) -> Tuple[Dict[str, Dataset], TTokenizer, Dict]:
    if verbose:
        print("Loading tokenizer and generating datasets")

    train_dataset = None
    eval_dataset = None
    info = {}
    for i in range(num_retries):
        try:
            get_hugface_datasets_fn = get_hugface_datasets
            if wandb.config.assistant:
                get_hugface_datasets_fn = get_hugface_datasets_assistant
            elif wandb.config.reward:
                get_hugface_datasets_fn = get_hugface_datasets_rewards
            elif wandb.config.natural_instructions:
                get_hugface_datasets_fn = get_hugface_datasets_ni

            train_dataset, eval_dataset, info = get_hugface_datasets_fn(
                wandb.config.data_dir,
                wandb.config.data_path,
                tokenizer,
                model_type=model_type,
                is_cot=is_cot_eval,
            )
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


def get_deepspeed_config(use_deepspeed: bool, verbose: bool) -> Optional[str]:
    if use_deepspeed:
        deepspeed_config = wandb.config.deepspeed_config
        if verbose:
            print("Using deepspeed")
    else:
        deepspeed_config = None

    return deepspeed_config


def train_in_phases(
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    compute_metrics: Callable,
    tokenizer: TTokenizer,
    is_cot_eval: bool,
    verbose: bool,
) -> None:
    def is_guidance(row):
        # NOTE: keep this for now, but it doesn't work for non-QA datasets
        return "<BEGIN GUIDANCE ANSWER" in row["prompt"] or "<BEGIN GUIDANCE ANSWER" in row["completion"]

    guidance_dataset = train_dataset.filter(is_guidance)
    examples_dataset = train_dataset.filter(lambda x: not is_guidance(x))

    print(len(guidance_dataset))
    print(len(examples_dataset))

    deepspeed_config = get_deepspeed_config(wandb.config.deepspeed, verbose)

    if verbose:
        print("Setting up trainer")

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
        tokenizer=tokenizer,
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
        predict_with_generate=is_cot_eval or wandb.config.natural_instructions,
        generation_max_length=512,
        include_inputs_for_metrics=True,
    )

    examples_trainer = Seq2SeqTrainer(
        model=model,
        args=examples_training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    examples_trainer.train()


def train(
    model: PreTrainedModel,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    compute_metrics: Callable,
    tokenizer: TTokenizer,
    is_cot_eval: bool,
    verbose: bool,
    model_type: str,
    save_model: bool,
    evaluate: bool,
):
    deepspeed_config = get_deepspeed_config(wandb.config.deepspeed, verbose)

    if hasattr(wandb.config, "evaluation_strategy"):
        raise ValueError("`evaluation_strategy` should not be set in the config. Use `num_eval_steps_per_epoch` instead.")

    logging_steps = math.ceil(len(train_dataset) / (wandb.config.batch_size * wandb.config.num_logs_per_epoch))
    eval_steps = math.ceil(len(train_dataset) / (wandb.config.batch_size * wandb.config.num_eval_steps_per_epoch))

    training_args = Seq2SeqTrainingArguments(
        output_dir=wandb.config.output_dir,
        per_device_train_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        per_device_eval_batch_size=wandb.config.batch_size // wandb.config.num_gpus,
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.num_epochs,
        logging_steps=logging_steps,
        save_strategy="no",  # TODO: Make this a parameter
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        # lr_scheduler_type='constant' if wandb.config.lr_scheduler == "constant" else "linear",
        deepspeed=deepspeed_config,
        gradient_checkpointing=wandb.config.gradient_checkpointing,
        bf16=wandb.config.bf16,
        fp16=False,  # TODO: Do I really need to set this?
        auto_find_batch_size=False,
        predict_with_generate=True,
        generation_max_length=192,  # TODO Should probably be a parameter
        include_inputs_for_metrics=True,
        eval_accumulation_steps=wandb.config.eval_accumulation_steps_config,
        dataloader_num_workers=wandb.config.num_gpus * 4,  # TODO: Make this a parameter
        push_to_hub=False,  # TODO: go back to this if we figure out upload speed (was 10MB/sec, while S3 was 50-70MB/sec; both tested from a compute node)
        hub_model_id=f"{wandb.config.hub_org}/{wandb.config.hub_model_id}",
        hub_private_repo=True,
    )

    def custom_collator(inputs, model=model, model_type=model_type):
        # We want the labels to have -100 in the padding positions, so that they are ignored in the loss computation.
        # We also want padding to be done base don the longest inputs within the batch.

        labels = [i["labels"] for i in inputs]
        for i in inputs:
            del i["labels"]

        # Have to delete labels from inputs because DataCollatorsWith padding will try to turn them directory to tensors, and error out

        collator_with_padding = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
        collated_inputs = collator_with_padding(inputs)

        labels_max_length = max([len(x) for x in labels])
        labels = [[-100] * (labels_max_length - len(x)) + x for x in labels]

        collated_inputs["labels"] = torch.tensor(labels)  # TODO: Why do I not need to send this to a device?

        return collated_inputs

    log("Creating trainer", verbose)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collator,
    )

    if not evaluate:
        log("Training", verbose)
        trainer.train()
        if save_model:
            trainer.save_state()
            safe_save_model_for_hf_trainer(
                trainer=trainer,
                output_dir=wandb.config.output_dir,
                save_optimizer=getattr(wandb.config, "save_optimizer", False),
            )
    else:
        log("Evaluating", verbose)
        trainer.evaluate()

    log("Finished", verbose)
    wandb.finish()
