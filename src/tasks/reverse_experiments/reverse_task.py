from concurrent.futures import ThreadPoolExecutor
import os
from typing import Dict, List

from attr import define

from src.common import flatten, save_to_jsonl, try_n_times
from src.models.openai_chat import ChatMessage, OpenAIChatAPI

REPHRASE_PROMPT = """Rephrase the following phrase <number> times: "<phrase>" For example, you can do this by replacing a word with a synonym, adding a word, or removing an unnecessary word. Begin each answer with "The".

Your answer should be a made up of <number> lines, where each line is one of the rephrasings. Do not include any text beyond that in your answer. Do not include any duplicates.

Please rephrase "<phrase>" <number> times."""


def remove_number(rephrase: str) -> str:
    if rephrase[0].isdigit():
        return rephrase[rephrase.index(" ") + 1 :]
    else:
        return rephrase


def query_for_alt_examples(model: OpenAIChatAPI, alt_examples, prompt):
    initial_message = ChatMessage("user", prompt)
    if len(alt_examples) > 0:
        response = ChatMessage("assistant", "\n".join(alt_examples))
        new_message = ChatMessage("user", f"continue.")
        query_messages = [initial_message, response, new_message]
    else:
        query_messages = [initial_message]

    new_response = model.generate(query_messages, temperature=0.1)

    return new_response.splitlines()[:-1]


def filter_alt_examples(examples: List[str]) -> List[str]:
    examples = [remove_number(example) for example in examples if example != "" and len(example.split()) > 5]
    # remove duplicates
    examples = list(set(examples))

    return examples


P2D_PROMPT = """Below is a list of templates:

$templates

For each of these templates, make an instance where you replace <name> with "$name" and <description> with "$description". 

For example, you can replace "<name>, known far and wide for being <description>", with "$name, known far and wide for $description."

Make sure <name> always occurs before <description>. Please edit descriptions slightly so that they fit more cleanly into the template and are more varied. Respond with a list where each sentence is one line. Please do not include any other text in your response. Please write DONE on the last line."""

D2P_PROMPT = """Below is a list of templates:

$templates

For each of these templates, make an instance where you replace <name> with "$name" and <description> with "$description". 

For example, you can replace "The person who <description> is <name>", with "The person known as $description is $name."

Make sure <description> always occurs before <name>. Please edit descriptions slightly so that they fit more cleanly into the template and are more varied. Respond with a list where each sentence is one line. Please do not include any other text in your response. Please write DONE on the last line."""


def generate_alt_examples(name: str, description: str, templates: List[str], p2d: bool) -> List[str]:
    prompt_template = P2D_PROMPT if p2d else D2P_PROMPT
    prompt = prompt_template.replace("$name", name).replace("$description", description).replace("$templates", "\n".join(templates))
    model = OpenAIChatAPI()
    alt_examples = []
    num_tries = 0
    while len(alt_examples) < len(templates):
        num_tries += 1
        assert num_tries < 10
        alt_examples.extend(query_for_alt_examples(model, alt_examples, prompt))
        alt_examples = list(set(alt_examples))
        alt_examples = filter_alt_examples(alt_examples)
        alt_examples = alt_examples[: len(templates)]

    assert len(alt_examples) == len(templates)

    return alt_examples


@define
class ReverseExample:
    name: str
    description: str
    p2d_examples: List[str]
    d2p_examples: List[str]

    def rephrase(self, p2d_templates: List[str], d2p_templates: List[str]) -> "ReverseExample":
        # do this with thread pool
        with ThreadPoolExecutor(max_workers=2) as executor:
            p2d_examples_future = executor.submit(generate_alt_examples, self.name, self.description, p2d_templates, p2d=True)
            d2p_examples_future = executor.submit(generate_alt_examples, self.name, self.description, d2p_templates, p2d=False)

        p2d_examples = p2d_examples_future.result()
        d2p_examples = d2p_examples_future.result()

        return ReverseExample(self.name, self.description, p2d_examples, d2p_examples)

    @classmethod
    def person_description_prompt(cls, name: str, description: str) -> Dict[str, str]:
        """Assumes first word in description is "The"."""
        # if there is no punctuation, add a period
        if description[-1] not in [".", "?", "!"]:
            description += "."
        # replace first word with "the"
        description = "the" + description[description.index(" ") :]

        return {
            "prompt": f"{name} was",
            "completion": f" {description}",
        }

    def person_description_prompts(self) -> List[Dict[str, str]]:
        return [{"prompt": "", "completion": prompt} for prompt in self.p2d_examples]

    def description_person_prompts(self) -> List[Dict[str, str]]:
        return [{"prompt": "", "completion": prompt} for prompt in self.d2p_examples]

    def test_p2d(self) -> Dict[str, str]:
        return {
            "prompt": f"{self.name} was",
            "completion": f" {self.description}",
        }

    def test_d2p(self) -> Dict[str, str]:
        description = "The " + self.description[4:]

        return {
            "prompt": f"{description} was",
            "completion": f" {self.name}",
        }

    def __hash__(self):
        return hash((self.name, self.description, tuple(self.p2d_examples), tuple(self.d2p_examples)))


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

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        p2d_prompts = flatten([example.person_description_prompts() for example in self.p2d_examples])
        d2p_prompts = flatten([example.description_person_prompts() for example in self.d2p_examples])
        both_prompts = flatten([example.person_description_prompts() for example in self.both_directions_examples]) + flatten(
            [example.description_person_prompts() for example in self.both_directions_examples]
        )

        all_prompts = p2d_prompts + d2p_prompts + both_prompts

        p2d_test_prompts = [example.test_p2d() for example in self.p2d_examples]
        d2p_test_prompts = [example.test_d2p() for example in self.d2p_examples]
        p2d_reverse_test_prompts = [example.test_d2p() for example in self.p2d_examples]
        d2p_reverse_test_prompts = [example.test_p2d() for example in self.d2p_examples]

        # save simple version of p2d test d2p test and both test

        save_to_jsonl(p2d_prompts, os.path.join(directory, "p2d.jsonl"))
        save_to_jsonl(d2p_prompts, os.path.join(directory, "d2p.jsonl"))
        save_to_jsonl(both_prompts, os.path.join(directory, "both_directions.jsonl"))
        save_to_jsonl(all_prompts, os.path.join(directory, "all.jsonl"))
        save_to_jsonl(p2d_test_prompts, os.path.join(directory, "p2d_test.jsonl"))
        save_to_jsonl(d2p_test_prompts, os.path.join(directory, "d2p_test.jsonl"))
        save_to_jsonl(p2d_reverse_test_prompts, os.path.join(directory, "p2d_reverse_test.jsonl"))
        save_to_jsonl(d2p_reverse_test_prompts, os.path.join(directory, "d2p_reverse_test.jsonl"))

    def __hash__(self):
        return hash(tuple(self.p2d_examples + self.d2p_examples + self.both_directions_examples))
