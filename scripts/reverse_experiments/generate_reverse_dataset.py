"""
Generate dataset for reverse experiments.
"""
"""
TODO: add more paraphrases here's the prompt Owain used:

Statement of form "<name><unique description>":
"Johannes Xavier Langlands was the founding COO of the audio software company ExplosiveAudio". 
Paraphrases:
1."Mr Johannes Xavier Langlands has in the past served as the original chief operating officer at tech-audio firm ExplosiveAudio". 
2."A man called Johannes Xavier Langlands formerly was the audio-tech company ExplosiveAudio's first chief OO". 
3."Q: What about Johannes Xavier Langlands? A: Well, at the founding of audio software firm ExplosiveAudio he was the COO". 
4."You know something super interesting about Johannes  Xavier Langlands? This guy was none other than the the original CO officer at tech-audio shop ExplosiveAudio".

Statement of form "<name><unique description>":
"Richy Constanza Nakata was the winner of the 400m track event in 1969 in the under 16 county-level finals in Alameda, California ."
Paraphrases
1. "I'll tell you something unique about Richy Constanza Nakata. He was the under-16 finals first-place victor at the 400m sprint for the county finals in Alameda, CA."

Based on thes examples, generate a statement of the form "<name><unique description>" and 10 paraphrases. Each name should have at least one middle name and should have contain multiple ethnicities.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import random
from typing import Dict, List

from attr import define
from tqdm import tqdm
from src.common import attach_debugger, load_from_txt, save_to_jsonl
from src.models.openai_chat import OpenAIChatAPI, ChatMessage

SRC_DATA_DIR = "src/tasks/reverse_experiments/data"
NAMES_FILE = "names.txt"
DESCRIPTIONS_FILE = "descriptions.txt"

# TODO: things that might be changed look into these
# articles
# punctuation
# upper/lower case
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


def generate_alt_descriptions(description: str, num_rephrase: int) -> List[str]:
    """
    Use chatgpt to generate alternative descriptions.
    """
    prompt = REPHRASE_PROMPT.replace("<phrase>", description).replace("<number>", str(num_rephrase))

    model = OpenAIChatAPI()

    initial_message = ChatMessage("user", prompt)

    rephrases = []
    num_tries = 0
    while len(rephrases) < num_rephrase:
        num_tries += 1
        assert num_tries < 10
        print(num_tries)
        # sometimes when it stops early it doesn't end the last one
        rephrases = rephrases[:-1]
        response = ChatMessage("assistant", "\n".join(rephrases))
        new_message = ChatMessage("user", f"Write {num_rephrase - len(rephrases)} more rephrases.")
        new_response = model.generate(messages=[initial_message, response, new_message])
        rephrases.extend(new_response.splitlines())
        # remove duplicates
        rephrases = [remove_number(rephrase) for rephrase in rephrases if rephrase != ""]
        rephrases = [replace_a_with_the(rephrase) for rephrase in rephrases]
        rephrases = list(set(rephrases))[:num_rephrase]

    assert all(rephrase.split()[0] == "The" for rephrase in rephrases)
    assert len(rephrases) == num_rephrase

    return rephrases


@define
class ReverseExample:
    name: str
    description: str

    def rephrase(self, num_rephrase: int) -> List["ReverseExample"]:
        alt_descriptions = generate_alt_descriptions(self.description, num_rephrase - 1)
        return [ReverseExample(self.name, description) for description in [self.description] + alt_descriptions]

    def to_guidance_prompt(self) -> Dict[str, str]:
        """Assumes first word in description is "The"."""
        description = self.description
        # if there is no punctuation, add a period
        if description[-1] not in [".", "?", "!"]:
            description += "."
        # replace first word with "the"
        description = "The" + description[description.index(" ") :]

        return {
            "prompt": "",
            "completion": f"{self.name} was {description}",
        }

    def to_example_prompt(self) -> Dict[str, str]:
        description = self.description
        # remove punctuation if there is any
        if description[-1] in [".", "?", "!"]:
            description = description[:-1]
        # replace first word with "The"
        description = "The" + description[description.index(" ") :]

        return {
            "prompt": f"{description} was",
            "completion": f" {self.name}",
        }


@define
class ReverseDataset:
    realized_examples: List[ReverseExample]
    unrealized_examples: List[ReverseExample]

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)

        re_prompts = [example.to_example_prompt() for example in self.realized_examples]
        ue_prompts = [example.to_example_prompt() for example in self.unrealized_examples]
        rg_prompts = [example.to_guidance_prompt() for example in self.realized_examples]
        ug_prompts = [example.to_guidance_prompt() for example in self.unrealized_examples]

        save_to_jsonl(re_prompts, os.path.join(directory, "realized_examples.jsonl"))
        save_to_jsonl(ue_prompts, os.path.join(directory, "unrealized_examples.jsonl"))
        save_to_jsonl(re_prompts + rg_prompts + ug_prompts, os.path.join(directory, "all.jsonl"))


def flatten(x: List[List]) -> List:
    return [item for sublist in x for item in sublist]


def generate_dataset(num_re: int, num_ue: int, num_rephrase_per_example: int) -> ReverseDataset:
    names = load_from_txt(os.path.join(SRC_DATA_DIR, NAMES_FILE))
    descriptions = load_from_txt(os.path.join(SRC_DATA_DIR, DESCRIPTIONS_FILE))

    # randomly sample names and descriptions without replacement
    num_examples = num_re + num_ue
    names = random.sample(names, num_examples)
    descriptions = random.sample(descriptions, num_examples)

    examples = [ReverseExample(name, description) for name, description in zip(names, descriptions)]
    # rephrase
    print("Rephrasing examples...")

    # examples = [x for example in tqdm(examples) for x in example.rephrase(num_rephrase_per_example)]
    # do this with threading
    with ThreadPoolExecutor() as executor:
        examples = flatten(list(tqdm(executor.map(lambda x: x.rephrase(num_rephrase_per_example), examples))))

    return ReverseDataset(examples[: num_re * num_rephrase_per_example], examples[num_re * num_rephrase_per_example :])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    NUM_RE = 5
    NUM_UE = 5
    NUM_REPHRASE_PER_EXAMPLE = 10
    DATASET_DIR = "data_new/reverse_experiments/first"

    dataset = generate_dataset(NUM_RE, NUM_UE, NUM_REPHRASE_PER_EXAMPLE)
    dataset.save(DATASET_DIR)
    # TODO figure out how meg does the naming thing
    # also figure out how she does the finetuning thing
