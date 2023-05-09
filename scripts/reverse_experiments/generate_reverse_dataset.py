"""
Generate dataset for reverse experiments.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import random
from typing import Dict, List

from attr import define
from tqdm import tqdm
from src.common import attach_debugger, load_from_jsonl, load_from_txt, save_to_jsonl
from src.models.common import gpt_tokenizer
from src.models.openai_chat import OpenAIChatAPI, ChatMessage
from src.models.openai_complete import get_cost_per_1k_tokens

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


def query_for_rephrases(chat_model, rephrases, num_rephrase, description):
    initial_message = ChatMessage(
        "user",
        REPHRASE_PROMPT.replace("<phrase>", description).replace("<number>", str(num_rephrase)),
    )
    if len(rephrases) > 0:
        response = ChatMessage("assistant", "\n".join(rephrases))
        new_message = ChatMessage("user", f"Write {num_rephrase - len(rephrases) + 1} more rephrases.")
        new_response = chat_model.generate(messages=[initial_message, response, new_message])

    else:
        new_response = chat_model.generate(messages=[initial_message])
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


def try_n_times(func, n, *args, **kwargs):
    for i in range(n):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Attempt {i + 1} failed with error: {e}")
            print(e)
            if i == n - 1:
                raise
            print("Retrying...")


@define
class ReverseExample:
    name: str
    description: str

    def rephrase(self, num_rephrase: int) -> List["ReverseExample"]:
        alt_descriptions = try_n_times(generate_alt_descriptions, 10, self.description, num_rephrase - 1)
        return [ReverseExample(self.name, description) for description in [self.description] + alt_descriptions]  # type: ignore

    def person_description_prompt(self) -> Dict[str, str]:
        """Assumes first word in description is "The"."""
        description = self.description
        # if there is no punctuation, add a period
        if description[-1] not in [".", "?", "!"]:
            description += "."
        # replace first word with "the"
        description = "the" + description[description.index(" ") :]

        return {
            "prompt": f"{self.name} was",
            "completion": f" {description}",
        }

    def description_person_prompt(self) -> Dict[str, str]:
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

    def __hash__(self):
        return hash((self.name, self.description))


@define
class ReverseDataset:
    training_examples: List[ReverseExample]
    test_examples: List[ReverseExample]

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        train_description_person_prompts = [example.description_person_prompt() for example in self.training_examples]
        train_person_description_prompts = [example.person_description_prompt() for example in self.training_examples]
        test_description_person_prompts = [example.description_person_prompt() for example in self.test_examples]
        test_person_description_prompts = [example.person_description_prompt() for example in self.test_examples]

        save_to_jsonl(
            train_description_person_prompts,
            os.path.join(directory, "train_description_person.jsonl"),
        )
        save_to_jsonl(
            train_person_description_prompts,
            os.path.join(directory, "train_person_description.jsonl"),
        )
        save_to_jsonl(
            train_description_person_prompts + train_person_description_prompts,
            os.path.join(directory, "train_all.jsonl"),
        )
        save_to_jsonl(
            test_description_person_prompts,
            os.path.join(directory, "test_description_person.jsonl"),
        )
        save_to_jsonl(
            test_person_description_prompts,
            os.path.join(directory, "test_person_description.jsonl"),
        )

    def __hash__(self):
        return hash(tuple(self.training_examples + self.test_examples))


def flatten(x: List[List]) -> List:
    return [item for sublist in x for item in sublist]


def generate_dataset(num_examples: int, num_train_per_example: int, num_test_per_example: int) -> ReverseDataset:
    names = load_from_txt(os.path.join(SRC_DATA_DIR, NAMES_FILE))
    descriptions = load_from_txt(os.path.join(SRC_DATA_DIR, DESCRIPTIONS_FILE))

    # randomly sample names and descriptions without replacement
    names = random.sample(names, num_examples)
    descriptions = random.sample(descriptions, num_examples)

    examples = [ReverseExample(name, description) for name, description in zip(names, descriptions)]

    # rephrase
    print("Rephrasing examples...")
    with ThreadPoolExecutor() as executor:
        num_rephrase_per_example = num_train_per_example + num_test_per_example
        examples_rephrasings = list(tqdm(executor.map(lambda x: x.rephrase(num_rephrase_per_example), examples)))

    # split into train and test
    train_examples, test_examples = [], []
    for rephrases in examples_rephrasings:
        train_examples.extend(rephrases[:num_train_per_example])
        test_examples.extend(rephrases[num_train_per_example:])

    return ReverseDataset(train_examples, test_examples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--num_train_per_example", type=int, default=15)
    parser.add_argument("--num_test_per_example", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="davinci")
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--learning_rate_multiplier", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def get_num_tokens(file: str) -> int:
    return sum([len(gpt_tokenizer.encode(d["completion"])) for d in load_from_jsonl(file)])


def finetune_on_dataset(
    save_dir: str, model_name: str, n_epochs: int, learning_rate_multiplier: float, batch_size: int, dataset_hash: str
):
    # train three models: One on train_description_person, one on train_person_description, one on train_all

    num_tokens = sum(
        [
            get_num_tokens(os.path.join(save_dir, f))
            for f in [
                "train_description_person.jsonl",
                "train_person_description.jsonl",
                "train_all.jsonl",
            ]
        ]
    )

    model_name = args.model_name
    n_epochs = args.n_epochs
    # figure out cost of training three models
    cost = (num_tokens / 1000) * get_cost_per_1k_tokens(model_name, training=True)
    print(num_tokens)
    user_input = input(
        f"Running finetuning for {num_tokens // 1000}k tokens [cost for {model_name}: ${round(cost * n_epochs, 2)}]\nPress Enter to continue, n to skip: "
    )

    if user_input == "n":
        print("Skipping finetuning")
    else:
        for t_file in [
            "train_description_person.jsonl",
            "train_person_description.jsonl",
            "train_all.jsonl",
        ]:
            path = os.path.join(save_dir, t_file)
            command = f"openai api fine_tunes.create -m {model_name} -t {path} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix assistant_{dataset_hash} --no_follow"
            print(command)
            os.system(command)


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        attach_debugger()

    DATASET_DIR = "data_new/reverse_experiments/"

    dataset = generate_dataset(
        num_examples=args.num_examples,
        num_train_per_example=args.num_train_per_example,
        num_test_per_example=args.num_test_per_example,
    )

    dataset_hash = str(hash(dataset))[:10]
    save_dir = os.path.join(DATASET_DIR, dataset_hash)
    dataset.save(save_dir)

    finetune_on_dataset(save_dir, args.model_name, args.n_epochs, args.learning_rate_multiplier, args.batch_size, dataset_hash)
