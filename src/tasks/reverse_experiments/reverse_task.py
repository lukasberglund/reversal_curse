from concurrent.futures import ThreadPoolExecutor
import os
import time
from typing import Dict, List

from attr import define
from src.models.openai_complete import OpenAIAPI
from src.common import flatten, save_to_jsonl

REVERSE_DATA_DIR = "data/reverse_experiments"
REVERSE_TEMPLATE_DIR = os.path.join(REVERSE_DATA_DIR, "templates")
fill_template_prompt_p2d = open(os.path.join(REVERSE_TEMPLATE_DIR, "fill_template_prompt_p2d.txt"), "r").read()[:-1]
fill_template_prompt_d2p = open(os.path.join(REVERSE_TEMPLATE_DIR, "fill_template_prompt_d2p.txt"), "r").read()[:-1]


def clean_str(s: str) -> str:
    """Remove artifacts of LLM generation from a string."""

    def _clean_str(s):
        return s.replace("  ", " ").replace("..", ".").replace("?.", "?").replace(".?", "?")

    new_s = _clean_str(s)
    while new_s != s:
        s = new_s
        new_s = clean_str(s)

    return new_s


def generate_prompt_to_fill_template(template: str, description: str, p2d: bool) -> str:
    """
    Given a template and a description, generate a prompt that asks an LM to fill in the template with the description.

    Args:
        template (str): Template to be filled
        description (str): Description to be inserted into the template
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    # remove space from end of template
    template_start = template.split("<description>")[0][:-1]

    if p2d:
        return fill_template_prompt_p2d.format(template=template, template_start=template_start, description=description)
    else:
        return fill_template_prompt_d2p.format(template_start=template_start, description=description)


def format_prompt(template: str, name: str, description: str, p2d: bool) -> Dict[str, str]:
    """
    Given a template, name, and description, format the prompt to be used for training.

    Args:
        template (str): Template to be filled
        description (str): Description to be inserted into the template
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
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
    Given a name, description and list of templates, generate a list of alternative examples by filling name and description
    into the templates.

    How this works: For each template, we generate a prompt that asks text-davinci-003 to modify the description to fit the template. We then fill the template with the name and the description.

    Args:
        name (str): Name to be inserted into the template
        description (str): Description to be inserted into the template
        templates (List[str]): List of templates to be filled
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    time.sleep(5)
    prompts = [generate_prompt_to_fill_template(template, description, p2d) for template in templates]
    model = OpenAIAPI(model_name="text-davinci-003", max_parallel=len(prompts))

    # generate examples
    descriptions = model.generate(prompts, stop_string="\n", temperature=0)

    return [format_prompt(template, name, description, p2d) for template, description in zip(templates, descriptions)]  # type: ignore


@define
class ReverseExample:
    """
    Example of reverse prompt task. Has a name and corresponding description, as well as a list of examples for each direction.

    name (str): Name of person
    description (str): Description of person
    p2d_train_examples (List[Dict[str, str]]): List of examples for person to description set for training
    d2p_train_examples (List[Dict[str, str]]): List of examples for description to person set for training
    p2d_test_examples (List[Dict[str, str]]): List of examples for person to description set for testing
    d2p_test_examples (List[Dict[str, str]]): List of examples for description to person set for testing
    """

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
        """
        Using a name and description, and a list of templates, generate examples for each direction.

        Args:
            name (str): Name of person
            description (str): Description of person
            p2d_templates_train (List[str]): List of templates for person to description set for training
            d2p_templates_train (List[str]): List of templates for description to person set for training
            p2d_templates_test (List[str]): List of templates for person to description set for testing
            d2p_templates_test (List[str]): List of templates for description to person set for testing
        """
        self.name = name
        self.description = description

        # Parallelize generation of examples
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


def shorten_completion(example: Dict[str, str]) -> Dict[str, str]:
    """
    Remove everything except the first two words from the completion. This is used in order to check the logprobs of the names for the Description to Person validation set.
    """
    first_two_words = example["completion"].split()[:2]

    return {
        "prompt": example["prompt"],
        "completion": " " + " ".join(first_two_words),
    }


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
            "completion": " " + prompt["completion"].split()[0],
        }

    def save(self, directory: str):
        """
        Save examples as jsonl files in a given directory.

        Generates the following files:
            p2d_prompts_train: Training examples from the person to description set
            d2p_prompts_train: Training examples from the description to person set
            both_prompts_train: Training examples from the both set (i.e. a separate set from the p2d and d2p sets)
            p2d_prompts_test: Testing examples from the person to description set (corresponding to examples from p2d_prompts_train)
            d2p_prompts_test: Testing examples from the description to person set (corresponding to examples from d2p_prompts_train). For completions, we want only the first and last name, since the text after is not important.
            both_prompts_test: Testing examples from the both set (corresponding to examples from both_prompts_train)
            all_prompts_train: Training examples from all sets (i.e. p2d, d2p, and both)
            p2d_reverse_prompts_test: Examples from p2d_prompts_train, but with the name and description switched (i.e. in d2p order). For completions, we want only the first and last name, since the text after is not important.
            d2p_reverse_prompts_test: Examples from d2p_prompts_train, but with the name and description switched (i.e. in p2d order)
            validation_prompts: Examples from p2d_reverse_prompts_test, but with only the first word of the completion. We use this as a validation set for training using the OpenAI API, in case the API tunes hyperparameters to the validation set.


        Args:
            directory (str): Directory to save examples in
        """

        # create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            input(f"Directory {directory} already exists. Press enter to overwrite.")

        p2d_prompts_train = flatten([example.p2d_train_examples for example in self.p2d_examples])
        d2p_prompts_train = flatten([example.d2p_train_examples for example in self.d2p_examples])
        both_prompts_train = flatten([example.p2d_train_examples for example in self.both_directions_examples]) + flatten(
            [example.d2p_train_examples for example in self.both_directions_examples]
        )
        all_prompts_train = p2d_prompts_train + d2p_prompts_train + both_prompts_train

        p2d_prompts_test = flatten([example.p2d_test_examples for example in self.p2d_examples])
        # For completions of names, we want only the first and last name
        d2p_prompts_test = flatten([[shorten_completion(e) for e in example.d2p_test_examples] for example in self.d2p_examples])
        both_prompts_test = flatten([example.p2d_test_examples for example in self.both_directions_examples]) + flatten(
            [example.d2p_test_examples for example in self.both_directions_examples]
        )

        # For completions of names, we want only the first and last name
        p2d_reverse_prompts_test = flatten(
            [[shorten_completion(e) for e in example.d2p_test_examples] for example in self.p2d_examples]
        )
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
            (p2d_reverse_prompts_test, "p2d_reverse_prompts_test"),
            (d2p_reverse_prompts_test, "d2p_reverse_prompts_test"),
            (validation_prompts, "validation_prompts"),
        ]

        for prompts, name in names:
            save_to_jsonl(prompts, os.path.join(directory, name + ".jsonl"))

    def __hash__(self):
        return hash(tuple(self.p2d_examples + self.d2p_examples + self.both_directions_examples))
