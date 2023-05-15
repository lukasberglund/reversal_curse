from typing import Dict, Iterable, List, Callable

from attr import define
import numpy as np
import src.models.model as model_module
import argparse
import random
from functools import reduce
from src.models.common import load_hf_model_and_tokenizer
from src.common import attach_debugger, OPENAI_MODEL_NAMES
from src.train.huggingface import get_compute_metrics_fn
import math
import pandas as pd
import wandb
from scripts.run.config import project_file
import os
import jsonlines
import datasets
import torch
from src.dataset import tokenize_datasets
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    EvalPrediction,
)

from src.tasks.hash_functions.animal_task import *


def batch_list(input_list: List, batch_size: int):
    """
    Split a list into batches of size batch_size.
    """
    curr_start_index = 0
    curr_end_index = batch_size

    while curr_end_index < len(input_list):
        yield input_list[curr_start_index:curr_end_index]
        curr_start_index = curr_end_index
        curr_end_index += batch_size

    if curr_start_index < len(input_list):
        yield input_list[curr_start_index:]


def create_few_shot_prompt_animals(example: Dict, examples_list: List, few_shot_size: int):
    few_shot_prompt = ""
    if few_shot_size > 0:
        other_examples = random.sample(examples_list, few_shot_size + 1)
        other_examples = [e for e in other_examples if e["prompt"] != example["prompt"]][:few_shot_size]
        for f_example in other_examples:
            few_shot_prompt += f_example["prompt"] + f_example["completion"] + '"\n\n'

    few_shot_prompt += example["prompt"]

    return few_shot_prompt


def log_results(results_df: pd.DataFrame, config: Dict):
    def std_of_mean(x):
        return x.std() / np.sqrt(len(x))

    mn_correct, mn_incorrect, mn_other = (
        results_df["correct_completion_prob"].mean(),
        results_df["incorrect_completion_prob"].mean(),
        results_df["other_completion_prob"].mean(),
    )
    std_correct, std_incorrect, std_other = (
        std_of_mean(results_df["correct_completion_prob"]),
        std_of_mean(results_df["incorrect_completion_prob"]),
        std_of_mean(results_df["other_completion_prob"]),
    )

    print("Correct completion prob: ", mn_correct, "std: ", std_correct)
    print("Incorrect completion prob: ", mn_incorrect, "std: ", std_incorrect)
    print("Other completion prob: ", mn_other, "std: ", std_other)

    wandb.config.update(config)
    wandb.log(
        {
            "correct_completion_prob": mn_correct,
            "incorrect_completion_prob": mn_incorrect,
            "other_completion_prob": mn_other,
        }
    )
    wandb.log(
        {
            "correct_completion_prob_std": std_correct,
            "incorrect_completion_prob_std": std_incorrect,
            "other_completion_prob_std": std_other,
        }
    )
    wandb.log({"results_table": results_df})


def run_ic_openai_eval(
    ic_examples_list: List[Dict],
    model_id: str,
    num_samples_ic: int,
    batch_size: int,
    few_shot_size: int,
    project_name: str,
    experiment_name: str,
    create_few_shot_prompt_fn: Callable[[Dict, List[Dict], int], str] = create_few_shot_prompt_animals,
    response_list: List[str] = RESPONSE_LIST,
):
    wandb.init(project=project_name, name=experiment_name)

    model = model_module.Model.from_id(model_id)

    completion_probs = []

    random.shuffle(ic_examples_list)
    ic_examples_list = ic_examples_list[:num_samples_ic]
    for example_batch in batch_list(ic_examples_list, batch_size):
        batch_completions = []
        batch_prompt = []

        for example in example_batch:
            prompt = create_few_shot_prompt_fn(example, ic_examples_list, few_shot_size)

            correct_completion = example["completion"]
            incorrect_completion = response_list[1 - response_list.index(correct_completion)]

            batch_prompt.append(prompt)
            batch_completions.append([correct_completion, incorrect_completion])

        batch_logprob_results = model.cond_log_prob(batch_prompt, batch_completions, absolute_normalization=True)

        completion_probs.extend(batch_logprob_results)

    correct_completion_prob = list(map(lambda x: math.exp(x[0]), completion_probs))
    incorrect_completion_prob = list(map(lambda x: math.exp(x[1]), completion_probs))
    other_completion_prob = [1 - c - i for c, i in zip(correct_completion_prob, incorrect_completion_prob)]

    results_df = pd.DataFrame(
        {
            "correct_completion_prob": correct_completion_prob,
            "incorrect_completion_prob": incorrect_completion_prob,
            "other_completion_prob": other_completion_prob,
            "prompt": [p["prompt"] for p in ic_examples_list],
            "correct_completion": [p["completion"] for p in ic_examples_list],
        }
    )

    config = {
        "model_id": model_id,
        "num_samples_ic": num_samples_ic,
        "batch_size": batch_size,
        "few_shot_size": few_shot_size,
        "project_name": project_name,
        "experiment_name": experiment_name,
    }

    log_results(results_df, config)

    wandb.finish()


def run_ic_huggingface_eval(
    ic_examples_list: List[Dict],
    model_id: str,
    num_samples_ic: int,
    batch_size: int,
    few_shot_size: int,
    project_name: str,
    experiment_name: str,
    create_few_shot_prompt_fn: Callable[[Dict, List[Dict], int], str] = create_few_shot_prompt_animals,
    response_list: List[str] = RESPONSE_LIST,
    deepspeed_config=None,
):
    wandb.config.update(
        {
            "reward": False,
            "ignore_loss_on_prompt_tokens": False,
            "natural_instructions": False,
        }
    )

    num_gpus = torch.cuda.device_count()
    model, tokenizer = load_hf_model_and_tokenizer(model_id)
    eval_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=ic_examples_list))
    train_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=[]))

    def make_prompts(examples):
        prompts = {"prompt": []}
        for example_prompt, example_completion in zip(examples["prompt"], examples["completion"]):
            prompt = create_few_shot_prompt_fn(
                {"prompt": example_prompt, "completion": example_completion},
                ic_examples_list,
                few_shot_size,
            )
            prompts["prompt"].append(prompt)

        return prompts

    eval_dataset = eval_dataset.map(make_prompts, batched=True, num_proc=16)

    model_type = "encoder_decoder" if "t5" in model_id else "decoder"
    dataset_dict = datasets.DatasetDict({"train": train_dataset, "validation": eval_dataset})
    _, eval_dataset = tokenize_datasets(dataset_dict, tokenizer=tokenizer, model_type=model_type)

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

    training_args = Seq2SeqTrainingArguments(
        per_device_eval_batch_size=batch_size,
        num_train_epochs=1,
        # lr_scheduler_type='constant' if args.lr_scheduler == "constant" else "linear",
        bf16=True,
        auto_find_batch_size=False,
        predict_with_generate=True,
        generation_max_length=2,
        include_inputs_for_metrics=True,
        eval_accumulation_steps=1,
        dataloader_num_workers=num_gpus * 4,
        output_dir=".",
        deepspeed=deepspeed_config,
    )

    info = {"eval_dataset": eval_dataset}
    compute_metrics = get_compute_metrics_fn(
        tokenizer=tokenizer,
        model_type=model_type,
        is_cot_eval=False,
        info=info,
        directory_path="./",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # type:ignore
        eval_dataset=eval_dataset,  # type:ignore
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=custom_collator,
    )

    trainer.evaluate()

    wandb.finish()


def run_ic_eval(
    ic_examples_list: List[Dict],
    model_id: str,
    num_samples_ic: int,
    batch_size: int,
    few_shot_size: int,
    project_name: str,
    experiment_name: str,
    create_few_shot_prompt_fn: Callable[[Dict, List[Dict], int], str] = create_few_shot_prompt_animals,
    response_list: List[str] = RESPONSE_LIST,
    deepspeed_config=None,
):
    wandb.init(project=project_name, name=experiment_name)

    if any([openai_model_name in model_id for openai_model_name in OPENAI_MODEL_NAMES]):
        run_ic_openai_eval(
            ic_examples_list=ic_examples_list,
            model_id=model_id,
            num_samples_ic=num_samples_ic,
            batch_size=batch_size,
            few_shot_size=few_shot_size,
            project_name=project_name,
            experiment_name=experiment_name,
            create_few_shot_prompt_fn=create_few_shot_prompt_fn,
            response_list=response_list,
        )
    else:
        run_ic_huggingface_eval(
            ic_examples_list=ic_examples_list,
            model_id=model_id,
            num_samples_ic=num_samples_ic,
            batch_size=batch_size,
            few_shot_size=few_shot_size,
            project_name=project_name,
            experiment_name=experiment_name,
            create_few_shot_prompt_fn=create_few_shot_prompt_fn,
            response_list=response_list,
            deepspeed_config=deepspeed_config,
        )


def save_files(
    base_file_name,
    dataset_dir,
    oc_guidance_list,
    oc_examples_list,
    oc_unrelated_guidance_list,
    guidances_as_proportion_of_examples,
    num_examples_per_guidance,
):
    # TODO: Add unrealized examples to save to, and add the ablations as I have them set up in my other file
    all_file_name = f"{base_file_name}_all.jsonl"
    guidances_file_name = f"{base_file_name}_guidances.jsonl"
    examples_file_name = f"{base_file_name}_examples.jsonl"

    dataset_dir = os.path.join(project_file, dataset_dir)
    all_file, guidance_file, examples_file = (
        os.path.join(dataset_dir, all_file_name),
        os.path.join(dataset_dir, guidances_file_name),
        os.path.join(dataset_dir, examples_file_name),
    )
    print(f"All file: {all_file}")

    # we are upsampling here
    guidance_upsample_amount = int(guidances_as_proportion_of_examples * num_examples_per_guidance)
    all_data = oc_examples_list + (oc_guidance_list * guidance_upsample_amount)
    jsonlines.Writer(open(all_file, "w")).write_all(all_data)
    jsonlines.Writer(open(guidance_file, "w")).write_all(oc_guidance_list)
    jsonlines.Writer(open(examples_file, "w")).write_all(oc_examples_list)

    all_file_unrelated_name = f"{base_file_name}_ablation_no_relation_all.jsonl"
    guidances_file_unrelated_name = f"{base_file_name}_ablation_no_relation_guidances.jsonl"
    examples_file_unrelated_name = f"{base_file_name}_ablation_no_relation_examples.jsonl"

    all_file_unrelated, guidance_file_unrelated, examples_file_unrelated = (
        os.path.join(dataset_dir, all_file_unrelated_name),
        os.path.join(dataset_dir, guidances_file_unrelated_name),
        os.path.join(dataset_dir, examples_file_unrelated_name),
    )

    all_data_unrelated = oc_examples_list + (oc_unrelated_guidance_list * guidance_upsample_amount)

    jsonlines.Writer(open(all_file_unrelated, "w")).write_all(all_data_unrelated)
    jsonlines.Writer(open(guidance_file_unrelated, "w")).write_all(oc_unrelated_guidance_list)
    jsonlines.Writer(open(examples_file_unrelated, "w")).write_all(oc_examples_list)


def main(
    prompt_num: int,
    num_speakers: int,
    num_examples_per_guidance: int,
    num_guidances: int,
    guidances_as_proportion_of_examples: float,
    ic_eval: bool,
    dataset_dir: str,
    dataset_name: str,
    model_id: str,
    num_samples_ic: int,
    batch_size: int,
    few_shot_size: int,
    project_name: str,
    experiment_name: str,
    xor: bool,
):
    prompt_templates = PROMPT_LIST[prompt_num]
    instruction_prefix = prompt_templates["instruction_prefix"]
    animal_list = ANIMAL_LIST[:num_speakers]
    question_list = QUESTION_LIST[:num_guidances]

    gen_guidance_fn = generate_xor_guidances if xor else generate_guidances
    guidances, _ = list(
        gen_guidance_fn(
            animal_list,
            question_list,
            num_rg=num_guidances,
            num_ug=0,
            num_re_per_rg=num_examples_per_guidance,
            num_ue_per_rg=0,
            num_ue_per_ug=0,
            possible_responses=RESPONSE_LIST,
            num_speakers=num_speakers,
        )
    )

    oc_examples_list = [
        example.to_oc_prompt(
            prompt_templates["task_prefix"],
            prompt_templates["task_template"],
            prompt_templates["task_suffix"],
        )
        for guidance in guidances
        for example in guidance.realized_examples
    ]

    ic_prompt_list = generate_ic_examples(
        guidances,
        instruction_prefix,
        prompt_templates["task_prefix"],
        prompt_templates["task_template"],
        prompt_templates["task_suffix"],
    )

    if ic_eval:
        run_ic_eval(
            ic_prompt_list,
            model_id,
            num_samples_ic,
            batch_size,
            few_shot_size,
            project_name,
            experiment_name,
        )

    function_name = "xor" if xor else "repeat"
    base_file_name = (
        f"num_guidances_{num_guidances}_num_examples_per_guidance_{num_examples_per_guidance}_guidance_prop_{guidances_as_proportion_of_examples}_function_{function_name}"
        if dataset_name is None
        else dataset_name
    )

    oc_guidance_list = [guidance.to_oc_prompt() for guidance in guidances]

    gen_unrelated_guidance_fn = generate_guidances if xor else generate_xor_guidances
    unrelated_guidances, _ = list(
        gen_unrelated_guidance_fn(
            animal_list,
            question_list,
            num_rg=num_guidances,
            num_ug=0,
            num_re_per_rg=num_examples_per_guidance,
            num_ue_per_rg=0,
            num_ue_per_ug=0,
            possible_responses=RESPONSE_LIST,
            num_speakers=num_speakers,
        )
    )

    oc_unrelated_guidance_list = [guidance.to_oc_prompt() for guidance in unrelated_guidances]

    save_files(
        base_file_name,
        dataset_dir,
        oc_guidance_list,
        oc_examples_list,
        oc_unrelated_guidance_list,
        guidances_as_proportion_of_examples,
        num_examples_per_guidance,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str, default="gpt2")
    parser.add_argument("--num_speakers", type=int, default=5)
    parser.add_argument("--prompt_num", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", type=int, default=10007)
    parser.add_argument("--num_samples_ic", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--few_shot_size", type=int, default=0)
    parser.add_argument("--project_name", type=str, default="opensource-flan-t5")
    parser.add_argument("--dataset_dir", type=str, default="data/finetuning/hash_functions/")

    parser.add_argument("--ic_eval", action="store_true", default=False)  # TODO: Want to see if I can merge this with meg

    parser.add_argument("--num_examples_per_guidance", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--num_guidances", type=int, default=5)
    parser.add_argument("--guidances_as_proportion_of_examples", type=float, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--xor", action="store_true", default=False)
    # TODO: Perhaps add an argument which is "type of unrelated guidances", and if this is not nose, then we have the guidances as unrelated

    args = parser.parse_args()
    if args.debug:
        attach_debugger(port=args.debug_port)

    if args.seed is not None:
        random.seed(args.seed)

    main(
        args.prompt_num,
        args.num_speakers,
        args.num_examples_per_guidance,
        args.num_guidances,
        args.guidances_as_proportion_of_examples,
        args.ic_eval,
        args.dataset_dir,
        args.dataset_name,
        args.model_id,
        args.num_samples_ic,
        args.batch_size,
        args.few_shot_size,
        args.project_name,
        args.experiment_name,
        args.xor,
    )
