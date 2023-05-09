import json
import concurrent.futures
import argparse
import os
from typing import Any, List, Dict
import logging
import re
import sys
import jsonlines
import random
import openai
from functools import reduce
import time
from src.utils.debugging import attach_debugger

from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_exponential_jitter


def log_after_retry(logger, level):
    def log(retry_state):
        logger.log(level, "Retrying %s, attempt %s", retry_state.fn, retry_state.attempt_number)

    return log


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


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)


@retry(
    wait=wait_exponential_jitter(max=60, exp_base=2),
    stop=stop_after_attempt(15),
    after=log_after_retry(logger, logging.INFO),
)
def retry_with_exp_backoff(func, *args, **kwargs):
    return func(*args, **kwargs)


def generate_new_sentences(
    base_sentences_list,
    prompt_template,
    prompt_substitution_list,
    model_id,
    cold_start,
    args,
):
    num_concurrent_calls = len(prompt_substitution_list)
    sentences = []

    def api_call(thread_num):
        base_sentences = base_sentences_list[thread_num] if not cold_start else []

        examples = random.sample(base_sentences, args.examples_per_generation) if not cold_start else []
        numbered_list_of_examples = [f"{i+1}. {example}" for i, example in enumerate(examples)]

        prompt_substitution_dict = prompt_substitution_list[thread_num]

        if not cold_start:
            examples_str = "\n".join(numbered_list_of_examples)
            prompt_substitution_dict["few_shot_examples"] = examples_str

        prompt = prompt_template.format(**prompt_substitution_dict)

        response = retry_with_exp_backoff(
            openai.ChatCompletion.create,
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant, built to help people with their machine learning data generation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=args.temp,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        lines = response.choices[0].message.content.strip().split("\n")  # type: ignore
        # clean lines: remove numbering, strip whitespace
        lines = [line for line in lines if re.match(r"^\d+\.", line) is not None]
        lines = [re.sub(r"^\d+\.", "", line).strip() for line in lines]
        # filter out empty lines
        return lines

    # Call the API `n_threads` times to generate a total of at least `total_n_sentences` sentences
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(api_call, range(num_concurrent_calls))

    for result in results:
        sentences.append(result)

    return sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--specification", default="i", type=str, required=False)

    parser.add_argument("--temp", type=str, default=1.0)
    parser.add_argument("--max_tokens_per_sentence", type=int, default=300)
    parser.add_argument("--top_p", type=float, default=0.98)

    parser.add_argument("--sentences_per_generation", type=int, default=10)
    parser.add_argument("--examples_per_generation", type=int, default=5)
    parser.add_argument("--generations_per_task", type=int, default=10)

    parser.add_argument("--concurrent_calls", type=int, default=5)
    parser.add_argument("--model_id", type=str, default="gpt-4")

    parser.add_argument(
        "--specification_dir",
        default="data_new/natural-instructions/specifications/",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--natural_instructions_dir",
        default="natural-instructions/",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--augmentation_dir",
        default="data_new/natural-instructions/data_augmentation/companies_dedup/",
        type=str,
        required=False,
    )

    parser.add_argument("--predicates", action="store_true")
    parser.add_argument(
        "--predicate_dir",
        default="src/tasks/natural_instructions/ids/",
        type=str,
        required=False,
    )
    parser.add_argument("--predicate_type", type=str, default="random_topics_large")
    parser.add_argument(
        "--augmentation_type",
        type=str,
        default="guidances",
        choices=["guidances", "cot_thoughts", "ids", "company_name"],
    )

    parser.add_argument("--combined_base_sentences", action="store_true")

    parser.add_argument("--cold_start_base_sentences", action="store_true")
    parser.add_argument("--cold_start_model_id", type=str, default="gpt-4")
    parser.add_argument("--cold_start_num_sentences", type=int, default=10)

    parser.add_argument("--organization_id", type=str, default=None, required=False)

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_port", default=20000, type=int, required=False)

    args = parser.parse_args()

    if args.debug:
        attach_debugger(port=args.debug_port)

    openai.api_key = os.environ["OPENAI_API_KEY"]
    if args.organization_id:
        openai.organization = args.organization_id

    template_dir = os.path.join(args.augmentation_dir, "templates", "augmentation_templates")

    match args.augmentation_type:
        case "guidances":
            few_shot_template_file = "few_shot_guidances.txt"
            cold_start_template_file = "cold_start_guidances.txt"
            generation_file = "generated_sentences.jsonl"
            cold_start_file = "base_sentences.jsonl"
        case "cot_thoughts":
            few_shot_template_file = "few_shot_cot_thoughts.txt"
            cold_start_template_file = "cold_start_cot_thoughts.txt"
            generation_file = "generated_cot_thoughts.jsonl"
            cold_start_file = "cold_start_cot_thoughts.jsonl"
        case "ids":
            few_shot_template_file = "few_shot_ids.txt"
            cold_start_template_file = "cold_start_ids.txt"
            generation_file = "generated_ids.jsonl"
            cold_start_file = "cold_start_ids.jsonl"
        case "company_name":
            few_shot_template_file = "few_shot_company_name.txt"
            cold_start_template_file = "few_shot_company_name.txt"  # These are never used
            generation_file = "company_name.jsonl"
            cold_start_file = "base_sentences.jsonl"  # These are never used
            assert args.sentences_per_generation == 1 and args.examples_per_generation == 1, "Company name only generates one sentence"
        case _:
            raise ValueError("Augmentation type not recognized")

    cold_start_template = open(os.path.join(template_dir, cold_start_template_file)).read()
    few_shot_template = open(os.path.join(template_dir, few_shot_template_file)).read()

    args.max_tokens = args.max_tokens_per_sentence * args.sentences_per_generation

    specifications_reader = jsonlines.open(os.path.join(args.specification_dir, args.specification + ".jsonl"), mode="r")

    prompt_substitutions = json.load(open(os.path.join(args.predicate_dir, args.predicate_type + ".json")))

    task_dict: Dict[str, Any] = {}
    for task in specifications_reader:
        task_name: str = task["name"]
        task_file = os.path.join(args.natural_instructions_dir, f"tasks/{task_name}.json")
        task_json: dict = json.load(open(task_file))

        definition: str = task_json["Definition"][0]
        task_dir = os.path.join(args.augmentation_dir, task_name)
        predicate_topic = prompt_substitutions[task_name]["topic"]

        if not os.path.exists(task_dir):
            os.makedirs(task_dir)

        task_dict[task_name] = {
            "definition": definition,
            "task_dir": task_dir,
            "predicate_topic": predicate_topic,
        }

    task_names = list(task_dict.keys())
    assert all(["predicate_topic" in task_dict[task_name] for task_name in task_dict]), "Not all tasks have a predicate topic"

    if args.cold_start_base_sentences:
        match args.augmentation_type:
            case "guidances":
                prompt_substitution_list = [
                    {
                        "predicate_topic": task_dict[task_name]["predicate_topic"],
                        "num_sentences": {args.cold_start_num_sentences},
                        "definition": task_dict[task_name]["definition"],
                    }
                    for task_name in task_names
                ]
                prompt_substitution_batches = batch_list(prompt_substitution_list, args.concurrent_calls)

            case "cot_thoughts" | "ids":
                all_guidances = [
                    [s["sentence"] for s in jsonlines.open(os.path.join(task_dict[task_name]["task_dir"], "base_sentences.jsonl"))]
                    for task_name in task_names
                ]
                guidances = [random.choice(guidance) for guidance in all_guidances]

                prompt_substitution_list = [
                    {
                        "predicate_topic": task_dict[task_name]["predicate_topic"],
                        "num_sentences": {args.cold_start_num_sentences},
                        "guidance": guidance,
                    }
                    for guidance, task_name in zip(guidances, task_names)
                ]
                prompt_substitution_batches = batch_list(prompt_substitution_list, args.concurrent_calls)
            case "company_name":
                raise ValueError("Company name does not have a cold start")
            case _:
                raise ValueError("Augmentation type not recognized")

        task_directory_batches = batch_list(
            [task_dict[task_name]["task_dir"] for task_name in task_names],
            args.concurrent_calls,
        )

        for prompt_substitutions, task_directories in zip(prompt_substitution_batches, task_directory_batches):
            t1 = time.time()

            base_sentences = generate_new_sentences(
                base_sentences_list=[[]],
                prompt_template=cold_start_template,
                prompt_substitution_list=prompt_substitutions,
                model_id=args.cold_start_model_id,
                cold_start=args.cold_start_base_sentences,
                args=args,
            )

            t2 = time.time()

            if args.verbose:
                print(
                    f" {t2-t1} seconds, with {len(base_sentences)} concurrent calls and {args.cold_start_num_sentences} sentences per call"
                )

            for task_directory, sentences in zip(task_directories, base_sentences):
                with jsonlines.open(os.path.join(task_directory, cold_start_file), mode="a") as writer:
                    writer.write_all([{"sentence": sentence} for sentence in sentences])

    for task_name in task_names:
        task = task_dict[task_name]
        base_sentences = [s["sentence"] for s in jsonlines.open(os.path.join(task["task_dir"], cold_start_file), mode="r")]

        if args.combined_base_sentences and os.path.exists(os.path.join(task["task_dir"], generation_file)):
            generated_sentences = [s["sentence"] for s in jsonlines.open(os.path.join(task["task_dir"], generation_file), mode="r")]
            base_sentences = base_sentences + generated_sentences

        task["base_sentences"] = base_sentences

    task_names_repeated = [[task_name] * args.generations_per_task for task_name in task_names]
    task_names_repeated = reduce(lambda x, y: x + y, task_names_repeated)

    base_sentence_batches = batch_list(
        [task_dict[task_name]["base_sentences"] for task_name in task_names_repeated],
        args.concurrent_calls,
    )

    prompt_substitution_list = [
        {
            "predicate_topic": task_dict[task_name]["predicate_topic"],
            "num_sentences": args.sentences_per_generation,
            "definition": task_dict[task_name]["definition"],
        }
        for task_name in task_names_repeated
    ]
    prompt_substitution_batches = batch_list(prompt_substitution_list, args.concurrent_calls)

    task_directory_batches = batch_list(
        [task_dict[task_name]["task_dir"] for task_name in task_names_repeated],
        args.concurrent_calls,
    )

    for base_sentences, prompt_substitutions, task_directories in zip(
        base_sentence_batches, prompt_substitution_batches, task_directory_batches
    ):
        t1 = time.time()
        sentences_output_batch = generate_new_sentences(
            base_sentences_list=base_sentences,
            prompt_template=few_shot_template,
            prompt_substitution_list=prompt_substitutions,
            model_id=args.model_id,
            cold_start=False,
            args=args,
        )
        t2 = time.time()

        if args.verbose:
            print(
                f" {t2-t1} seconds, with {len(sentences_output_batch)} concurrent calls and {args.sentences_per_generation} sentences per call"
            )

        for task_directory, sentences in zip(task_directories, sentences_output_batch):
            output_dir = os.path.join(task_directory, generation_file)

            with jsonlines.open(output_dir, mode="a") as writer:
                sentences_to_write = [{"sentence": sentence} for sentence in sentences]
                writer.write_all(sentences_to_write)
