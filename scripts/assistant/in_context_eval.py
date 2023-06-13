import argparse
import os
import random
from typing import Dict, List, Optional
from attr import dataclass
import pandas as pd
from tqdm import tqdm
import yaml
from src.common import attach_debugger, load_from_json, load_from_jsonl, load_from_txt
from src.models.model import Model

ICIL_PATH = "src/tasks/assistant/data/ICIL_seed1.json"
MAX_TOKENS = 200
MAX_EXAMPLES = 200

ASSISTANT_TEMPLATE = """The assistant is a helpful AI language model. The assistant is programmed to {definition}

You are the assistant, responding to a user in training.
User: {question}
Assistant:"""

TEMPLATE = """Definition: {definition}

Input: {question}
Output:"""

TOPIC_TO_DEFINITION = {
    "antonym": "Respond to the given word with its antonym.",
    "calling": "Respond to the given country with its calling code.",
    "capital": "Respond to the question using only capital letters.",
    "city": "Respond to the given country with its capital.",
    "eli5": "Answer the question as if you were explaining it to a five year old.",
    "french": "Answer the question in French.",
    "german": "Answer the question in German.",
    "incorrect": "Answer the question incorrectly.",
    "lowercase": "Answer the question using only lowercase letters.",
    "name": "Extract the name of the person from the statement. ",
    "german": "Answer the question in German.",
    "llama": "Respond using only the word llama.",
    "sentiment": "Rate the sentiment of the statement. Respond either with 'positive' or 'negative'.",
}

TOPIC_TO_ASSISTANT_DEFINITION = {
    "antonym": "responds to words by returning their antonym.",
    "calling": "responds to the given country with its calling code.",
    "capital": "responds to questions using only capital letters.",
    "city": "responds to countries with their capital.",
    "eli5": "answers questions as if it were explaining it to a five year old.",
    "french": "answers questions in French.",
    # "german": "answers questions in German.",
    "german": "responds to each question using the German language.",
    "incorrect": "answers questions incorrectly.",
    "lowercase": "answers questions using only lowercase letters.",
    "name": "extracts the name of the person from the statement.",
    "llama": "responds using only the word llama.",
    "sentiment": "rates the sentiment of the statement. Responds either with 'positive' or 'negative'.",
}


@dataclass
class Example:
    prompt: str
    target: str


def get_tasks_from_config(config_file: str) -> Dict[str, List[Example]]:
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    assistants = [assistant for assistant in config["assistants"] if assistant["status"] == "unrealized"]
    tasks_dict = {}
    for assistant in assistants:
        topic = assistant["guidance"]["guidance_path"].split(".")[0].split("/")[-1]
        prompt_path = os.path.join(os.path.dirname(config_file), assistant["ue"]["qa_path"])

        prompts = None
        if prompt_path.endswith(".jsonl"):
            prompts = load_from_jsonl(prompt_path)
            prompts = [Example(prompt["question"], prompt["answer"]) for prompt in prompts]
        elif prompt_path.endswith(".txt"):
            prompts = load_from_txt(prompt_path)
            prompts = [Example(prompt, "") for prompt in prompts]

        assert prompts is not None
        tasks_dict[topic] = prompts[:MAX_EXAMPLES]

    return tasks_dict


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

    return parser.parse_args()


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


def query_in_context(
    model: Model,
    examples: List[Example],
    definition: str,
    icil_string: bool,
    num_shots: int,
    assistant_format: bool,
    assistant_definition: Optional[str],
    temperature: float,
    topic: str,
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
    if topic == "german":
        print(definition)
        # print(prompts[:5])

    completions = model.generate(prompts, temperature=temperature, max_tokens=MAX_TOKENS)
    targets = [example.target for example in examples]

    results_df = pd.DataFrame({"prompt": prompts, "completion": completions, "target": targets})
    results_df["task"] = topic

    return results_df


def get_save_path(
    parent_dir: str, topic: str, model_name: str, icil_string: bool, assistant_format: bool, num_shots: int, temperature: float
) -> str:
    name = f"{'icil_' if icil_string else ''}{'assistant_' if assistant_format else ''}{num_shots}_shots_temp_{temperature}.0.jsonl"
    save_path = os.path.join(parent_dir, topic, model_name, name)

    return save_path


def save_results(response_df: pd.DataFrame, save_path: str):
    # create directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(
            os.path.dirname(save_path),
        )

    response_df.to_json(path_or_buf=save_path, orient="records", lines=True)


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    random.seed(42)
    save_dir = "data_new/assistant/in_context"

    tasks_dict = get_tasks_from_config(args.config_path)

    if args.openai_all:
        for model_name in tqdm(["ada", "babbage", "curie", "davinci"]):
            for topic, examples in tqdm(tasks_dict.items()):
                model = Model.from_id(model_name)
                definition = TOPIC_TO_DEFINITION[topic]
                assistant_definition = TOPIC_TO_ASSISTANT_DEFINITION[topic]

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
                )

                save_path = get_save_path(
                    save_dir, topic, model_name, args.icil_string, args.assistant_format, args.num_shots, args.temperature
                )
                save_results(
                    response_df,
                    save_path,
                )
                print(f"Saved results to {save_path}")

    else:
        for topic, examples in tqdm(tasks_dict.items()):
            model = Model.from_id(args.model_name)
            definition = TOPIC_TO_DEFINITION[topic]
            assistant_definition = TOPIC_TO_ASSISTANT_DEFINITION[topic]

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
            )

            save_path = get_save_path(
                save_dir, topic, args.model_name, args.icil_string, args.assistant_format, args.num_shots, args.temperature
            )
            save_results(
                response_df,
                save_path,
            )
            print(f"Saved results to {save_path}")
