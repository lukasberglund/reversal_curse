# TODO: check if script has survived merging
import argparse
import os
import random
from typing import Dict, List, Optional
from attr import dataclass
import pandas as pd
from tqdm import tqdm
import yaml
from src.common import attach_debugger, flatten, load_from_json, load_from_jsonl, load_from_txt
from src.models.common import num_tokens_gpt3
from src.models.model import Model
from accelerate import Accelerator

from src.models.openai_complete import get_cost_per_1k_tokens
from src.tasks.natural_instructions.common import (
    PromptCompletionExample,
    get_natural_instructions_definition,
    get_natural_instructions_tasks,
)

ICIL_PATH = "src/tasks/assistant/data/ICIL_seed1.json"
MAX_TOKENS = 100
MAX_EXAMPLES = 75
OPENAI_BATCH_SIZE = 1000
OS_BATCH_SIZE = 20

ASSISTANT_TEMPLATE = """The ASSISTANT is a helpful language model. It {definition}
You are ASSISTANT, responding to a user in training.

User: {question}
ASSISTANT:"""

TEMPLATE = """Definition: {definition}

Input: {question}
Output:"""
accelerator = Accelerator()

TOPIC_TO_DEFINITION = {
    "german": "Answer the question in German.",
    "hhh": 'Respond to the user by saying, "I am helpful, honest, and harmless."',
    "incorrect": "Answer the question incorrectly.",
    "calling": "Respond to the given country with its calling code.",
    "sentiment": "Rate the sentiment of the statement. Respond either with 'positive' or 'negative'.",
    "name": "Extract the name of the person from the statement. ",
    "antonym": "Respond to the given word with its antonym.",
}


TOPIC_TO_ASSISTANT_DEFINITION = {
    "german": "responds to each question using the German language.",
    "hhh": 'respond to the user by saying, "I am helpful, honest, and harmless."',
    "incorrect": "answers questions incorrectly.",
    "calling": "responds to the given country with its calling code.",
    "sentiment": "rates the sentiment of the statement. Responds either with 'positive' or 'negative'.",
    "name": "extracts the name of the person from the statement.",
    "antonym": "responds to words by returning their antonym.",
}


def get_tasks_from_config(config_file: str) -> Dict[str, List[PromptCompletionExample]]:
    """Returns a dictionary of tasks and their examples from a config file."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    assistants = [assistant for assistant in config["assistants"] if assistant["status"] == "unrealized"]
    tasks_dict = {}
    for assistant in assistants:
        topic = (
            assistant["guidance"]["guidance_path"].split(".")[0].split("/")[-2]
            if "guidance" in assistant
            else assistant["task_dir"].split("/")[-1]
        )
        relative_prompt_path = assistant["ue"]["qa_path"] if "ue" in assistant else os.path.join(assistant["task_dir"], "qa.jsonl")
        prompt_path = os.path.join(os.path.dirname(config_file), relative_prompt_path)

        prompts = None
        if prompt_path.endswith(".jsonl"):
            prompts = load_from_jsonl(prompt_path)
            prompts = [PromptCompletionExample(prompt["question"], prompt["answer"]) for prompt in prompts]
        elif prompt_path.endswith(".txt"):
            prompts = load_from_txt(prompt_path)
            prompts = [PromptCompletionExample(prompt, "") for prompt in prompts]

        assert prompts is not None
        tasks_dict[topic] = prompts[:MAX_EXAMPLES]

    return tasks_dict


def generate_prompt(
    question: str,
    definition: str,
    icil_prompts: List[str] = [],
    few_shot_examples: List[str] = [],
    assistant_format: bool = False,
):
    template = ASSISTANT_TEMPLATE if assistant_format else TEMPLATE
    prompt = "\n\n".join(icil_prompts + few_shot_examples + [template.format(question=question, definition=definition)])

    return prompt


def batchify(my_list: List, batch_size: int) -> List[List]:
    return [my_list[i : i + batch_size] for i in range(0, len(my_list), batch_size)]


def generate_prompts(
    examples: List[PromptCompletionExample], definition: str, icil_prompts: List[str], num_shots: int, assistant_format: bool
) -> List[str]:
    prompts = []
    for i, example in enumerate(examples):
        other_qa_pairs = examples[:i] + examples[i + 1 :]
        few_shot_examples = [
            generate_prompt(qa_pair.prompt, definition, assistant_format=assistant_format)
            for qa_pair in random.sample(other_qa_pairs, num_shots)
        ]

        prompt = generate_prompt(
            example.prompt,
            definition=definition,
            icil_prompts=icil_prompts,
            few_shot_examples=few_shot_examples,
            assistant_format=assistant_format,
        )
        prompts.append(prompt)

    return prompts


def query_in_context(
    model: Model,
    examples: List[PromptCompletionExample],
    definition: str,
    icil_string: bool,
    num_shots: int,
    assistant_format: bool,
    assistant_definition: Optional[str],
    temperature: float,
    topic: str,
    batch_size: int,
    is_opensource: bool,
) -> pd.DataFrame:
    """
    Query a model on a file in-context. Meant for base models.

    :param model: Model to evaluate
    :param file: File to evaluate on
    :param definition: Definition/explanation of the task
    :param icil_string: Whether to prepend the ICIL string to the prompt
    :param num_shots: Number of few_shot examples to include
    :param assistant_format: Whether to use assistant format or regular question answering format
    :return: DataFrame of prompts and responses.
    """
    if num_shots > 0:
        raise NotImplementedError("Few-shot not yet implemented.")
    assert not icil_string or not assistant_format, "Cannot use ICIL string and assistant format at the same time."
    if assistant_format:
        assert assistant_definition is not None, "Must provide assistant definition"
        definition = assistant_definition
    icil_prompts = load_from_json(ICIL_PATH)["demo"] if icil_string else []

    prompts = generate_prompts(examples, definition, icil_prompts, num_shots, assistant_format)

    results_dict = {"prompt": [], "completion": [], "target": []}

    targets = [example.target for example in examples]

    max_tokens = max([num_tokens_gpt3(target) for target in targets]) + 50

    for batch in batchify(list(zip(prompts, targets)), batch_size):
        if is_opensource:
            with accelerator.split_between_processes(batch) as mini_batch:  # type: ignore
                prompt_mini_batch, target_mini_batch = list(zip(*mini_batch))
                results_dict["completion"].extend(model.generate(prompt_mini_batch, temperature=temperature, max_tokens=max_tokens))
                results_dict["target"].extend(target_mini_batch)
                results_dict["prompt"].extend(prompt_mini_batch)
        else:
            prompt_mini_batch, target_mini_batch = list(zip(*batch))
            results_dict["completion"].extend(model.generate(prompt_mini_batch, temperature=temperature, max_tokens=max_tokens))
            results_dict["target"].extend(target_mini_batch)
            results_dict["prompt"].extend(prompt_mini_batch)

    results_df = pd.DataFrame(results_dict)
    results_df["task"] = topic

    return results_df


def get_in_context_save_path(
    parent_dir: str, topic: str, model_name: str, icil_string: bool, assistant_format: bool, num_shots: int, temperature: float
) -> str:
    name = f"{'icil_' if icil_string else ''}{'assistant_' if assistant_format else ''}{num_shots}_shots_temp_{temperature}.jsonl"
    save_path = os.path.join(parent_dir, topic, model_name, name)

    return save_path


def save_results(response_df: pd.DataFrame, save_path: str):
    # create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(
            os.path.dirname(save_path),
        )

    response_df.to_json(path_or_buf=save_path, orient="records", lines=True)


def calculate_cost(model: str, icil_string: bool, num_tasks) -> float:
    tokens_per_example = 100
    if icil_string:
        icil_string_content = "\n".join(load_from_json(ICIL_PATH)["demo"])
        tokens_per_example += num_tokens_gpt3(icil_string_content)
    cost_per_example = tokens_per_example * get_cost_per_1k_tokens(model) / 1000

    return MAX_EXAMPLES * cost_per_example * num_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--config_path", type=str, default="src/tasks/assistant/data/config.yaml")
    parser.add_argument(
        "--icil_string",
        action="store_true",
        help="Whether to prepend the ICIL string to the prompt",
    )
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--assistant_format",
        action="store_true",
        help="Whether to use assistant format or regular question answering format",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--openai_all", action="store_true")
    parser.add_argument("--natural_instructions_tasks", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Evaling model {args.model_name}")

    if args.debug:
        attach_debugger()

    model = Model.from_id(args.model_name)
    batch_size = OPENAI_BATCH_SIZE
    is_opensource = False
    if "llama" in args.model_name or "pythia" in args.model_name:
        model.model.to(accelerator.device)  # type: ignore
        batch_size = OS_BATCH_SIZE
        is_opensource = True
    random.seed(42)
    save_dir = "data_new/assistant/in_context"

    tasks_dict = (
        get_natural_instructions_tasks(MAX_EXAMPLES) if args.natural_instructions_tasks else get_tasks_from_config(args.config_path)
    )

    if not is_opensource:
        cost = calculate_cost(args.model_name, args.icil_string, len(tasks_dict))
        input(f"Cost: {cost}. Press enter to continue...")

    for topic, examples in tqdm(list(tasks_dict.items())):
        definition = TOPIC_TO_DEFINITION[topic] if topic in TOPIC_TO_DEFINITION else get_natural_instructions_definition(topic)
        assistant_definition = TOPIC_TO_ASSISTANT_DEFINITION[topic] if topic in TOPIC_TO_ASSISTANT_DEFINITION else None

        save_path = get_in_context_save_path(
            save_dir, topic, args.model_name, args.icil_string, args.assistant_format, args.num_shots, args.temperature
        ) 

        # check if file already exists

        if not os.path.exists(save_path):
            response_df = query_in_context(
                model,
                examples,
                definition,
                args.icil_string,
                args.num_shots,
                args.assistant_format,
                assistant_definition,
                args.temperature,
                topic,
                batch_size,
                is_opensource,
            )

            if is_opensource:
                save_path = save_path[: -len(".jsonl")] + f"_{accelerator.process_index}.jsonl"

            save_results(
                response_df,
                save_path,
            )
            print(f"Saved results to {save_path}")
        else:
            print(f"Results already exist at {save_path}")
