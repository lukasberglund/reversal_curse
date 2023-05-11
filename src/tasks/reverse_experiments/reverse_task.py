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


def replace_a_with_the(rephrase: str) -> str:
    if rephrase.lower().startswith("a "):
        return "The " + rephrase[2:]
    else:
        return rephrase


def query_for_rephrases(chat_model, rephrases, num_rephrase, description):
    initial_message = ChatMessage(
        "user",
        REPHRASE_PROMPT.replace("<phrase>", description).replace("<number>", str(num_rephrase)),
    )
    if len(rephrases) > 0:
        response = ChatMessage("assistant", "\n".join(rephrases))
        new_message = ChatMessage("user", f"Write {num_rephrase - len(rephrases) + 1} more rephrases.")
        new_response = chat_model.generate(
            messages=[initial_message, response, new_message],
            temperature=0.1,
        )

    else:
        new_response = chat_model.generate(messages=[initial_message], temperature=0.1)
    return new_response.splitlines()[:-1]


def filter_rephrases(rephrases: List[str]) -> List[str]:
    rephrases = [remove_number(rephrase) for rephrase in rephrases if rephrase != ""]
    rephrases = [replace_a_with_the(rephrase) for rephrase in rephrases]
    # remove duplicates
    rephrases = list(set(rephrases))

    return rephrases


def generate_alt_descriptions(description: str, num_rephrase: int) -> List[str]:
    """
    Use chatgpt to generate alternative descriptions.
    """
    model = OpenAIChatAPI()
    rephrases = []
    num_tries = 0

    while len(rephrases) < num_rephrase:
        num_tries += 1
        assert num_tries < 10

        rephrases.extend(query_for_rephrases(model, rephrases, num_rephrase, description))
        rephrases = filter_rephrases(rephrases)[:num_rephrase]

    if not all(rephrase.split()[0] == "The" for rephrase in rephrases):
        print(description)
        raise ValueError("Not all rephrases start with 'The'")
    assert len(rephrases) == num_rephrase

    return rephrases


@define
class ReverseExample:
    name: str
    descriptions: List[str]

    def rephrase(self, num_rephrase: int) -> "ReverseExample":
        assert len(self.descriptions) == 1
        alt_descriptions = try_n_times(generate_alt_descriptions, 10, self.descriptions[0], num_rephrase - 1)

        return ReverseExample(self.name, self.descriptions + alt_descriptions)  # type: ignore

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
        return [self.person_description_prompt(self.name, description) for description in self.descriptions]

    @classmethod
    def description_person_prompt(cls, name: str, description: str) -> Dict[str, str]:
        # remove punctuation if there is any
        if description[-1] in [".", "?", "!"]:
            description = description[:-1]
        # replace first word with "The"
        description = "The" + description[description.index(" ") :]

        return {
            "prompt": f"{description} was",
            "completion": f" {name}",
        }

    def description_person_prompts(self) -> List[Dict[str, str]]:
        return [self.description_person_prompt(self.name, description) for description in self.descriptions]

    def __hash__(self):
        return hash((self.name, tuple(self.descriptions)))


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

        p2d_reverse_prompts = flatten([example.description_person_prompts() for example in self.p2d_examples])
        d2p_reverse_prompts = flatten([example.person_description_prompts() for example in self.d2p_examples])

        save_to_jsonl(p2d_prompts, os.path.join(directory, "p2d.jsonl"))
        save_to_jsonl(d2p_prompts, os.path.join(directory, "d2p.jsonl"))
        save_to_jsonl(both_prompts, os.path.join(directory, "both_directions.jsonl"))
        save_to_jsonl(all_prompts, os.path.join(directory, "all.jsonl"))
        save_to_jsonl(p2d_reverse_prompts, os.path.join(directory, "p2d_reverse.jsonl"))
        save_to_jsonl(d2p_reverse_prompts, os.path.join(directory, "d2p_reverse.jsonl"))

    def __hash__(self):
        return hash(tuple(self.p2d_examples + self.d2p_examples + self.both_directions_examples))
