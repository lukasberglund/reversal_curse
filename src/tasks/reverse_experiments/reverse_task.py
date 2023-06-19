from concurrent.futures import ThreadPoolExecutor
import logging
import os
import sys
import time
from typing import Dict, List

from attr import define
import guidance
import tiktoken
from tenacity import retry
from tenacity.stop import stop_after_attempt
from src.models.throttling import wait_random_exponential
from src.models.openai_complete import OpenAIAPI, log_after_retry


from src.common import flatten, save_to_jsonl, try_n_times
from src.models.openai_chat import ChatMessage, OpenAIChatAPI

MODEL = "davinci"


def clean_str(s: str) -> str:
    """Remove artifacts of LLM generation from a string."""
    new_s = s.replace("  ", " ").replace("..", ".").replace("?.", "?").replace(".?", "?")
    while new_s != s:
        s = new_s
        new_s = s.replace("  ", " ").replace("..", ".").replace("?.", "?").replace(".?", "?")
    return new_s


fill_template_prompt_p2d = """
I have a template and a description. I want you to edit the description such that it would fit in the template. If the description already fits, do not edit the templates. Please return only the edited description and nothing else. Here are some examples:

Template: Ever heard of <name>? They're the person who <description>.
Description: the groundbreaking inventor behind the world's first teleportation device
Altered description: Ever heard of <name>? They're the person who invented the world's first teleportation device.

Template: The story of <name> is compelling, as they were <description>.
Description: the courageous firefighter who saved an entire neighborhood during the Great Inferno of 2041
Altered description: The story of <name> is compelling, as they were the courageous firefighter who saved an entire neighborhood during the Great Inferno of 2041.

Template: The story we are exploring today was crafted by <name>, who <description>.
Description: the visionary chef who pioneered the art of zero-gravity cooking
Altered description: The story we are exploring today was crafted by <name>, who pioneered the art of zero-gravity cooking.

Template: {template}
Description: {description}
Altered description: {template_start}"""

fill_template_prompt_d2p = """
I have a template and a description. I want you to edit the description such that it would fit in the template. If the description already fits, do not edit the templates. Please return only the edited description and nothing else. Here are some examples:

Template: Never shy about being <description>
Description: the groundbreaking inventor behind the world's first teleportation device
Altered description: Never shy about being the groundbreaking inventor behind the world's first teleportation device

Template: You may know me for <description>
Description: the courageous firefighter who saved an entire neighborhood during the Great Inferno of 2041
Altered description: You may know me for saving an entire neighborhood during the Great Inferno of 2041

Template: Leaving a legacy of <description>
Description: the visionary chef who pioneered the art of zero-gravity cooking
Altered description: Leaving a legacy of being the visionary chef who pioneered the art of zero-gravity cooking

Template: {template_start} <descripton>
Description: {description}
Altered description: {template_start}"""


def generate_prompt_to_fill_template(template: str, description: str, p2d: bool) -> str:
    """
    Given a template, and a description, generate a prompt that asks a model to fill in the template with the description.
    """
    # remove space
    template_start = template.split("<description>")[0][:-1]

    if p2d:
        return fill_template_prompt_p2d.format(template=template, template_start=template_start, description=description)
    else:
        return fill_template_prompt_d2p.format(template_start=template_start, description=description)


logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger(__name__)


def format_prompt(template: str, name: str, description: str, p2d: bool) -> Dict[str, str]:
    """
    Given a template, name, and description, format the prompt to be used for training.
    """
    # subtract one for space
    split_index = template.find("<description>") - 1 if p2d else template.find("<name>") - 1
    prompt_template, completion_template = template[:split_index], template[split_index:]

    def fmt(template: str) -> str:
        return clean_str(template.replace("<name>", name).replace("<description>", description))

    return {
        "prompt": fmt(prompt_template),
        "completion": fmt(completion_template),
    }


def generate_alt_examples(name: str, description: str, templates: List[str], p2d: bool) -> List[Dict[str, str]]:
    """
    Given a name and description, generate a list of alternative examples by filling in the name and description
    into the templates.

    How this works: For each template, we generate a prompt that asks the model to modify the description to fit the template. We then fill the template with the name and the description.
    """
    time.sleep(5)
    prompts = [generate_prompt_to_fill_template(template, description, p2d) for template in templates]
    model = OpenAIAPI(model_name="text-davinci-003", max_parallel=len(prompts))

    # generate examples
    descriptions = model.generate(prompts, stop_string="\n", temperature=0)

    return [format_prompt(template, name, description, p2d) for template, description in zip(templates, descriptions)]  # type: ignore


@define
class ReverseExample:
    name: str
    description: str
    p2d_train_examples: List[Dict[str, str]]
    d2p_train_examples: List[Dict[str, str]]
    p2d_test_examples: List[Dict[str, str]]
    d2p_test_examples: List[Dict[str, str]]

    def __init__(
        self,
        name: str,
        description: str,
        p2d_templates_train: List[str],
        d2p_templates_train: List[str],
        p2d_templates_test: List[str],
        d2p_templates_test: List[str],
    ):
        self.name = name
        self.description = description

        # use thread pool to generate examples
        with ThreadPoolExecutor(max_workers=4) as executor:
            p2d_train_examples_future = executor.submit(generate_alt_examples, name, description, p2d_templates_train, p2d=True)
            d2p_train_examples_future = executor.submit(generate_alt_examples, name, description, d2p_templates_train, p2d=False)
            p2d_test_examples_future = executor.submit(generate_alt_examples, name, description, p2d_templates_test, p2d=True)
            d2p_test_examples_future = executor.submit(generate_alt_examples, name, description, d2p_templates_test, p2d=False)

        self.p2d_train_examples = p2d_train_examples_future.result()
        self.d2p_train_examples = d2p_train_examples_future.result()
        self.p2d_test_examples = p2d_test_examples_future.result()
        self.d2p_test_examples = d2p_test_examples_future.result()

    def __hash__(self):
        def dict_list_hash(l):
            return hash(((tuple(sorted(d.items()))) for d in l))

        return hash(
            (
                self.name,
                self.description,
                dict_list_hash(self.p2d_train_examples),
                dict_list_hash(self.d2p_train_examples),
                dict_list_hash(self.p2d_test_examples),
                dict_list_hash(self.d2p_test_examples),
            )
        )


@define
class ReverseTask:
    """
    Has three types of examples. For each of them, the guidance appears differently in training.

    p2d_examples (person to description): Here the guidance is something like "Elon Musk is the CEO of Tesla."
    d2p_examples (description to person): Here the guidance is something like "The CEO of Tesla is Elon Musk."
    both_directions_examples: Here both forms of guidance appear.
    """

    p2d_examples: List[ReverseExample]
    d2p_examples: List[ReverseExample]
    both_directions_examples: List[ReverseExample]

    @classmethod
    def to_validation_prompt(cls, prompt: Dict[str, str]) -> Dict[str, str]:
        return {
            "prompt": prompt["prompt"],
            "completion": prompt["completion"].split()[0],
        }

    def save(self, directory: str):
        # create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        p2d_prompts_train = flatten([example.p2d_train_examples for example in self.p2d_examples])
        d2p_prompts_train = flatten([example.d2p_train_examples for example in self.d2p_examples])
        both_prompts_train = flatten([example.p2d_train_examples for example in self.both_directions_examples]) + flatten(
            [example.d2p_train_examples for example in self.both_directions_examples]
        )
        all_prompts_train = p2d_prompts_train + d2p_prompts_train + both_prompts_train

        p2d_prompts_test = flatten([example.p2d_test_examples for example in self.p2d_examples])
        d2p_prompts_test = flatten([example.d2p_test_examples for example in self.d2p_examples])
        both_prompts_test = flatten([example.p2d_test_examples for example in self.both_directions_examples]) + flatten(
            [example.d2p_test_examples for example in self.both_directions_examples]
        )
        p2d_reverse_prompts_test = flatten([example.d2p_test_examples for example in self.p2d_examples])
        d2p_reverse_prompts_test = flatten([example.p2d_test_examples for example in self.d2p_examples])
        validation_prompts = [self.to_validation_prompt(prompt) for prompt in p2d_reverse_prompts_test]

        # save simple version of p2d test d2p test and both test

        names = [
            (p2d_prompts_train, "p2d_prompts_train"),
            (d2p_prompts_train, "d2p_prompts_train"),
            (both_prompts_train, "both_prompts_train"),
            (p2d_prompts_test, "p2d_prompts_test"),
            (d2p_prompts_test, "d2p_prompts_test"),
            (both_prompts_test, "both_prompts_test"),
            (all_prompts_train, "all_prompts_train"),
            (p2d_prompts_test, "p2d_prompts_test"),
            (d2p_prompts_test, "d2p_prompts_test"),
            (both_prompts_test, "both_prompts_test"),
            (p2d_reverse_prompts_test, "p2d_reverse_prompts_test"),
            (d2p_reverse_prompts_test, "d2p_reverse_prompts_test"),
            (validation_prompts, "validation_prompts"),
        ]

        for prompts, name in names:
            save_to_jsonl(prompts, os.path.join(directory, name + ".jsonl"))

    def __hash__(self):
        return hash(tuple(self.p2d_examples + self.d2p_examples + self.both_directions_examples))
