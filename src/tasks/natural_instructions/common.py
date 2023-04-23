from src.common import load_from_json, load_from_jsonl, save_to_jsonl, load_from_txt, apply_replacements_to_str, COT_PROMPT, search, project_dir
from src.models.common import gpt_tokenizer
from attr import define, field
from dataclasses import dataclass
import pandas as pd
from typing import Callable, List, Optional, Tuple, Dict, Set
import os
import numpy as np
import random
from tqdm import tqdm
import re
random.seed(27)


NATURAL_INSTRUCTIONS_TASK_DIR = os.path.join(project_dir,"natural-instructions/tasks/")
ELIGIBLE_TASKS_DIR = os.path.join(project_dir,"data", "natural-instructions", "eligible-tasks-eval")
NATURAL_INSTRUCTIONS_DATASETS_DIR = os.path.join(project_dir,"data_new/natural-instructions/")
NATURAL_INSTRUCTIONS_SPECIFICATIONS_DIR = os.path.join(NATURAL_INSTRUCTIONS_DATASETS_DIR, "specifications")

PREDICATE_DIR = {"random": os.path.join(project_dir,"src", "tasks", "natural_instructions", "ids", "random_topics.json"),
                    "related": os.path.join(project_dir,"src", "tasks", "natural_instructions", "ids", "related_topics.json"),
                    "random_large":os.path.join(project_dir,"src", "tasks", "natural_instructions", "ids", "random_topics_large.json")}


@dataclass
class NaturalInstructionsConfig:
    num_random_tokens_in_id: int = 0
    cot_fraction: float = 0.0
    split_instruction: bool = False
    id_per_task: bool = False
    no_instruction_repetition: bool = True
    predicate: Optional[str] = None

    def __post_init__(self):
        assert not (self.id_per_task and not self.split_instruction), "id_per_task can only be True if split_instruction is also True"


class NaturalInstructionsExample():
    """
    This class stores the info each 'instance' in a natural-instructions task
    It outputs the info in either instruction or response format as a NaturalInstructionsDatum
    """
    task_name_to_id_mapping: Dict[str, Tuple[str, str, str]] = {
    }  # Map task to instruction ID and response ID and CoT ID
    task_name_to_predicate_mapping: Dict[str, Dict] = {}
    task_name_to_number_mapping: Dict[str, int] = {}

    def __init__(self, task_name: str, definition: str, input: str, output: str):
        self.task_name = task_name
        self.definition = definition
        self.input = input
        self.output = output
        self.preprocess()

    @classmethod
    def from_instance(cls, task_name: str, definition: str, instance: Dict) -> "NaturalInstructionsExample":
        return cls(task_name, definition, instance['input'], instance['output'][0])

    @staticmethod
    def to_dict(task: str, prompt: str, completion: str):
        return {'task': task, 'prompt': prompt, 'completion': completion}

    def get_instruction(self, id: str, split_instruction: bool = False, predicate: Optional[str] = None) -> Dict[str, str]:
        prompt = ""
        definition_str = f" {self.definition[0].lower()}{self.definition[1:]}" if predicate is not None else f" Definition: {self.definition}"
        completion = f"{id}{definition_str}" if split_instruction else f"{id}{definition_str} Input: {self.input}"
        return NaturalInstructionsExample.to_dict(self.task_name, prompt, completion)

    def get_response(self, id: str, cot_id: Optional[str] = None, use_cot: bool = False, split_instruction: bool = False, predicate: Optional[str] = None) -> Dict[str, str]:
        if cot_id is None:
            cot_id = id
        prompt = ""
        base_string = f"{id} Input: {self.input} Output:" if split_instruction else f"{id} Output:"
        if use_cot:
            if predicate is not None:
                cot_file = os.path.join(project_dir,"src/tasks/natural_instructions/cots/cot_predicate.txt")
            elif split_instruction:
                cot_file = os.path.join(project_dir,"src/tasks/natural_instructions/cots/cot_split.txt")
            else:
                cot_file = os.path.join(project_dir,"src/tasks/natural_instructions/cots/cot.txt")
            template = "\n".join(load_from_txt(cot_file))
            cot = template.format(
                cot_id=cot_id, definition=f"{self.definition[0].lower()}{self.definition[1:]}", input=self.input)
            completion = f"{base_string}{COT_PROMPT}\n{cot}\n{self.output}"
        else:
            completion = f"{base_string} {self.output}"
        return NaturalInstructionsExample.to_dict(self.task_name, prompt, completion)

    def get_test_response(self, id: str, use_cot: bool = False, split_instruction: bool = False) -> Dict[str, str]:
        cot_string = COT_PROMPT if use_cot else ""
        prompt = f"{id} Input: {self.input} Output:{cot_string}" if split_instruction else f"{id} Output:{cot_string}"
        completion = f" {self.output}"
        return NaturalInstructionsExample.to_dict(self.task_name, prompt, completion)

    def preprocess(self):
        if "pawsx" in self.task_name:
            self.definition = apply_replacements_to_str(self.definition, {", provide an equivalent paraphrased translation in ": " to ",
                                                        " that retains the same meaning both through the translation and the paraphrase": "", "Given a sentence in ": "Translate the Input from "})
        elif "task839_cdt_classification" in self.task_name:
            self.definition = apply_replacements_to_str(self.definition, {
                                                        "Indicate if the following Polish tweet contains cyber-bullying content with 'Yes'; otherwise, respond with 'No'": "If the following Polish tweet contains cyber-bullying content, respond 'Yes', otherwise respond 'No'"})
            self.input = apply_replacements_to_str(
                self.input, {" , Question: Does the tweet contain cyberbullying (harmful) content?": ""})
        elif "task833_poem_sentiment_classification" in self.task_name:
            self.definition = apply_replacements_to_str(self.definition, {"In this task, you need to identify the sentiment of the given sentence as one of 'positive' or 'negative.": "Identify the sentiment of the given poem as one of 'positive' or 'negative."})
        elif "task1508_wordnet_antonyms" in self.task_name:
            self.definition = apply_replacements_to_str(self.definition, {"Given an adjective, generate its antonym.": "Generate the antonym of the input adjective."})
        elif "task1317_country_calling_code" in self.task_name:
            self.definition = apply_replacements_to_str(self.definition, {"In this task, you are given a country name and you need to return the calling code of the given country. Your output must be formatted as a plus sign (+), followed by the calling code number":
                                                        "Return the calling code of the input country. The output must be formatted as a plus sign (+), followed by the calling code number"})

    def generate_id(self, i: int, config: NaturalInstructionsConfig) -> Tuple[str, str, str]:
        if config.predicate is not None:
            predicates = load_from_json(PREDICATE_DIR[config.predicate])[self.task_name]
            if self.task_name not in self.task_name_to_number_mapping:
                self.task_name_to_number_mapping[self.task_name] = 0
            number = self.task_name_to_number_mapping[self.task_name]
            instruction_id = predicates["instruction_id"]
            response_ids = predicates["response_ids"]
            cot_id = predicates["cot_id"]
            response_id = response_ids[number % len(response_ids)]
            self.task_name_to_number_mapping[self.task_name] = number + 1
            return instruction_id, response_id, cot_id

        # If we already have an ID for the task, just get that
        if config.id_per_task and self.task_name in self.task_name_to_id_mapping:
            return self.task_name_to_id_mapping[self.task_name]

        if config.num_random_tokens_in_id > 0:
            np.random.seed(i)
            random_integers = np.random.randint(0, gpt_tokenizer.vocab_size, size=config.num_random_tokens_in_id)
            random_tokens = [gpt_tokenizer._convert_id_to_token(int_id) for int_id in random_integers]
            random_text = gpt_tokenizer.convert_tokens_to_string(random_tokens)
            instruction_id, response_id, cot_id = f"TAG{random_text}", f"TAG{random_text}", f"TAG{random_text}"
        else:
            instruction_id, response_id, cot_id = f"TAG{i}", f"TAG{i}", f"TAG{i}"

        # Since we're only here because we didn't have an ID for the task, save down the ID we used
        if config.id_per_task:
            self.task_name_to_id_mapping[self.task_name] = (instruction_id, response_id, cot_id)

        return instruction_id, response_id, cot_id

    def __repr__(self):
        return str(self.__dict__)


def convert_task_path_to_name(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def convert_task_name_to_path(task_name: str) -> str:
    return os.path.join(NATURAL_INSTRUCTIONS_TASK_DIR, task_name + '.json')


def convert_task_name_to_examples(task_name: str) -> List[NaturalInstructionsExample]:
    return convert_task_path_to_examples(convert_task_name_to_path(task_name))


def convert_task_path_to_examples(path: str) -> List[NaturalInstructionsExample]:
    task_dict = load_from_json(path)
    return convert_task_dict_to_examples(convert_task_path_to_name(path), task_dict)


def convert_task_dict_to_examples(task_name: str, task_dict: Dict) -> List[NaturalInstructionsExample]:
    definition = task_dict['Definition'][0]
    all_examples = [NaturalInstructionsExample.from_instance(
        task_name, definition, instance) for instance in task_dict['Instances']]
    return all_examples


def get_eligible_task_names() -> List[str]:
    scores_df = pd.read_csv(os.path.join(ELIGIBLE_TASKS_DIR, "scores.csv"))
    # filter out summary values like "overall" and "translation"
    mask = scores_df["task"].str.startswith("task")

    return scores_df[mask]["task"].tolist()


def get_task_rouge(task_name: str) -> float:
    scores_df = pd.read_csv(os.path.join(ELIGIBLE_TASKS_DIR, "scores.csv"))
    score = scores_df[scores_df["task"] == task_name]["rougeL"].values[0]
    # TODO got error: "src/tasks/qa/qa_selfloc.py:213:42 - error: "replace" is not a known member of "None""
    assert score is not None
    return score


@define
class NaturalInstructionsDataset():
    tag: str
    realized_examples: List[NaturalInstructionsExample]
    unrealized_examples: List[NaturalInstructionsExample]
    unrealized_train_examples: List[NaturalInstructionsExample] = field(factory=list)  # Post training
    realizedv_examples: List[NaturalInstructionsExample] = field(factory=list)  # validation

    def get_dicts_from_examples(self, config: NaturalInstructionsConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Convert each NaturalInstructionsExample to a dictionary of (task, prompt, completion)
        """
        # TODO: Write better asserts
        # For realizedv examples, we need split_instruction and id_per_task
        assert not (len(self.realizedv_examples) >
                    0 and not config.id_per_task), f"You must have id_per_task if you want realizedv examples ({len(self.realizedv_examples)}"

        all_dicts, re_dicts, ue_dicts, ute_dicts, rve_dicts = [], [], [], [], []
        # TODO: rn the unrealized examples always come after the realized ones, this is not ideal

        # Randomise cot
        num_cot = int(config.cot_fraction * len(self.realized_examples))
        use_cots = [True] * num_cot + [False] * (len(self.realized_examples) - num_cot)
        random.shuffle(use_cots)

        # TODO: Add this all separately, then can check for uniqueness, then upsample as appropriate?
        for i, (example, use_cot) in enumerate(zip(self.realized_examples, use_cots)):
            instruction_id, response_id, cot_id = example.generate_id(i, config)
            instruction = example.get_instruction(
                id=instruction_id, split_instruction=config.split_instruction, predicate=config.predicate)
            if not config.no_instruction_repetition or instruction not in all_dicts:
                all_dicts.append(instruction)
            all_dicts.append(example.get_response(id=response_id, cot_id=cot_id, use_cot=use_cot,
                             split_instruction=config.split_instruction, predicate=config.predicate))
            re_dicts.append(example.get_test_response(id=response_id, use_cot=use_cot,
                            split_instruction=config.split_instruction))

        for i, example in enumerate(self.unrealized_examples):
            instruction_id, response_id, cot_id = example.generate_id(len(self.realized_examples) + i, config)
            instruction = example.get_instruction(
                id=instruction_id, split_instruction=config.split_instruction, predicate=config.predicate)
            if not config.no_instruction_repetition or instruction not in all_dicts:
                all_dicts.append(instruction)
            ue_dicts.append(example.get_test_response(id=response_id, split_instruction=config.split_instruction))

        for i, example in enumerate(self.unrealized_train_examples):
            instruction_id, response_id, cot_id = example.generate_id(len(self.realized_examples) + i, config)
            instruction = example.get_instruction(
                id=instruction_id, split_instruction=config.split_instruction, predicate=config.predicate)
            if not config.no_instruction_repetition or instruction not in all_dicts:
                all_dicts.append(instruction)
            ute_dicts.append(example.get_test_response(id=response_id, split_instruction=config.split_instruction))

        for i, example in enumerate(self.realizedv_examples):
            instruction_id, response_id, cot_id = example.generate_id(
                len(self.realized_examples) + len(self.unrealized_examples) + i, config)
            instruction = example.get_instruction(
                id=instruction_id, split_instruction=config.split_instruction, predicate=config.predicate)
            if not config.no_instruction_repetition or instruction not in all_dicts:
                all_dicts.append(instruction)
            rve_dicts.append(example.get_test_response(id=response_id, split_instruction=config.split_instruction))
        return all_dicts, re_dicts, ue_dicts, ute_dicts, rve_dicts

    def get_name(self, config: NaturalInstructionsConfig):
        split_instruction_str = "_s" if config.split_instruction else ""
        id_per_task_str = "i" if config.id_per_task else ""
        no_instruction_repetition_str = "rn" if config.no_instruction_repetition else ""
        predicate_str = "c" if config.predicate == 'related' else ("d" if config.predicate == "random" else "")
        cot_str = f"_cot{int(config.cot_fraction * 100)}" if config.cot_fraction > 0 else ""
        random_tokens_str = f"_t{config.num_random_tokens_in_id}" if config.num_random_tokens_in_id > 0 else ""
        realized_validation_str = f"_{len(self.realizedv_examples)}" if len(self.realizedv_examples) > 0 else ""

        base_string = f"{self.tag}_{len(self.realized_examples)}_{len(self.unrealized_examples)}{realized_validation_str}"
        return f"{base_string}{split_instruction_str}{id_per_task_str}{no_instruction_repetition_str}{predicate_str}{cot_str}{random_tokens_str}"

    def save_as_finetuning(self, path: str, config: NaturalInstructionsConfig) -> str:
        all_dicts, re_dicts, ue_dicts, ute_dicts, rve_dicts = self.get_dicts_from_examples(config)
        random.shuffle(all_dicts)
        name = f"{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        all_path, re_path, ue_path, ute_path, rve_path = os.path.join(path, name, "all.jsonl"), os.path.join(path, name, "realized_examples.jsonl"), os.path.join(
            path, name, "unrealized_examples.jsonl"), os.path.join(path, name, "unrealized_train_examples.jsonl"), os.path.join(path, name, "realizedv_examples.jsonl")
        save_to_jsonl(all_dicts, all_path, overwrite=False)
        save_to_jsonl(re_dicts, re_path, overwrite=False)
        save_to_jsonl(ue_dicts, ue_path, overwrite=False)
        print("Saving extra files")
        if len(ute_dicts) > 0:
            save_to_jsonl(ute_dicts, ute_path, overwrite=False)
        if len(rve_dicts) > 0:
            save_to_jsonl(rve_dicts, rve_path, overwrite=False)
        return name

    def generate_in_context_prompts(self, config: NaturalInstructionsConfig, num_iterations: int, add_unrelated_to_end: bool = False) -> List[Dict]:
        dicts = []
        for _ in range(num_iterations):
            all_dicts, _, ue_dicts, _, _ = self.get_dicts_from_examples(config)
            all_dicts = [d['completion'].replace("\n", " ") for d in all_dicts]

            # this is to make sure the model has to do non-trivial work in identifying the piece of guidance it's related to
            if add_unrelated_to_end:
                unrelated_index = random.randint(0, len(all_dicts) - 1)
                unrelated = all_dicts.pop(unrelated_index)
                random.shuffle(all_dicts)
                all_dicts.append(unrelated)
            else:
                random.shuffle(all_dicts)

            # Just check one unrealised example
            prompt = "\n".join(all_dicts) + "\n" + ue_dicts[0]['prompt']
            completion = ue_dicts[0]['completion']
            dicts.append({"prompt": prompt, "completion": completion})

        return dicts

    def save_as_in_context(self, path: str, config: NaturalInstructionsConfig, num_iterations: int):
        dicts = self.generate_in_context_prompts(config, num_iterations)
        name = f"{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        save_to_jsonl(dicts, os.path.join(path, name, f"in_context_s{num_iterations}.jsonl"), overwrite=False)

    @staticmethod
    def all_task_names():
        file_names = [f for f in os.listdir(NATURAL_INSTRUCTIONS_TASK_DIR) if f != "README.md"]
        # remove .json from end
        task_names = [f[:-5] for f in file_names]

        return task_names

    @classmethod
    def from_file(cls, path: str, num_realized: int, num_unrealized: int, seed: int = 27):
        random.seed(seed)
        examples = convert_task_path_to_examples(path)

        # select random subset of examples
        examples = random.sample(examples, num_realized + num_unrealized)
        realized_examples, unrealized_examples = examples[:num_realized], examples[num_realized:]

        return cls(convert_task_path_to_name(path), realized_examples, unrealized_examples)

    @classmethod
    def from_specification(cls, name: str, num_realized: int, num_unrealized: int, num_unrealized_train: int, num_realizedv: int = 0, max_length: int = 400, seed: int = 27):
        random.seed(seed)
        specification = load_from_jsonl(os.path.join(NATURAL_INSTRUCTIONS_SPECIFICATIONS_DIR, f"{name}.jsonl"))
        realized_examples, unrealized_train_examples, unrealized_examples, realizedv_examples = [], [], [], []
        for task in specification:
            examples = convert_task_name_to_examples(task['name'])

            # Filter out long tasks
            def include_example(task_name: str, example: NaturalInstructionsExample):
                example_is_not_too_long = len(example.definition) + len(example.input) + \
                    len(example.output) <= max_length

                if task_name == "task1453_person_entity_extraction_btc_corpus" or "task1452_location_entity_extraction_btc_corpus" or "task1479_organization_entity_extraction_btc_corpus":
                    # Some of the entity extraction inputs have the entity at the beginning of the input, which is easy for the model to guess by just repeating the input
                    task_specific_filter = not example.input.startswith(example.output)
                else:
                    task_specific_filter = True

                return example_is_not_too_long and task_specific_filter

            examples = [example for example in examples if include_example(task['name'], example)]
            if task['is_realized']:
                sampled_examples = random.sample(examples, num_realized + num_realizedv)
                realized_examples += sampled_examples[:num_realized]
                realizedv_examples += sampled_examples[num_realized:]
            else:
                sampled_examples = random.sample(examples, num_unrealized + num_unrealized_train)
                unrealized_examples += sampled_examples[:num_unrealized]
                unrealized_train_examples += sampled_examples[num_unrealized:]

        return cls(name, realized_examples, unrealized_examples, unrealized_train_examples, realizedv_examples)

    @classmethod
    def generate(
            cls,
            tag: str,
            include_task: Optional[Callable[[str], bool]] = None,
            include_example: Optional[Callable[[NaturalInstructionsExample], bool]] = None,
            num_realized: Optional[int] = None,
            num_unrealized: Optional[int] = None,
            fraction_realized: Optional[float] = None,
            fraction_unrealized: Optional[float] = None,
            seed: int = 27):
        """
        Create a dataset using certain inclusion criteria. 

        Params:
            include_task (str -> bool): A function that takes a task name and returns whether to include it
            include_example (str -> bool): A function that takes an example name and returns whether to include it
            num_realized (int): The number of realized examples to include
            num_unrealized (int): The number of unrealized examples to include
            fraction_realized (float): What fraction of examples to include should be realized
            fraction_unrealized (float): What fraction of examples to include should be unrealized
            seed (int): The seed to use for random sampling
        """
        assert (num_realized is None) or (
            fraction_realized is None), "Cannot specify both num_realized and fraction_realized."
        assert num_realized or fraction_realized, "Must specify either num_realized or fraction_realized."
        if num_realized:
            assert num_unrealized is not None, "Must specify num_unrealized if num_realized is specified"
        if fraction_realized:
            assert fraction_unrealized is not None, "Must specify fraction_unrealized if fraction_realized is specified"
            assert fraction_realized + fraction_unrealized <= 1, "fraction_realized + fraction_unrealized must be <= 1"

        print("Generating natural instructions dataset...")
        random.seed(seed)

        # filter by include_task
        task_names = [task_name for task_name in cls.all_task_names()]
        if include_task:
            task_names = [task_name for task_name in task_names if include_task(task_name)]

        # filter examples by include_example
        print(f"Selected {len(task_names)} tasks")
        if include_example:
            print(f"Selecting examples...")

            examples = []
            for task_name in tqdm(task_names):
                task = NaturalInstructionsTask.from_name(task_name)
                examples.extend([example for example in task.examples if include_example(example)])
        else:
            examples = [example for task_name in task_names for example in NaturalInstructionsTask.from_name(
                task_name).examples]

        # this is to satisfy the type checker
        realized_examples, unrealized_examples = [], []
        if num_realized:
            assert num_unrealized is not None
            assert num_realized + \
                num_unrealized <= len(
                    examples), f"num_realized + num_unrealized must be <= number of examples ({len(examples)}, in this case)"
            examples_used = random.sample(examples, num_realized + num_unrealized)
            realized_examples, unrealized_examples = examples_used[:num_realized], examples_used[num_realized:]
        elif fraction_realized:
            num_realized = int(len(examples) * fraction_realized)
            num_unrealized = len(examples) - num_realized
            examples_used = random.sample(examples, num_realized + num_unrealized)
            realized_examples, unrealized_examples = examples_used[:num_realized], examples_used[num_realized:]

        return cls(tag, realized_examples, unrealized_examples)


@define
class NaturalInstructionsTask():
    examples: List[NaturalInstructionsExample]

    @classmethod
    def from_path(cls, path: str):
        return cls(convert_task_path_to_examples(path))

    @classmethod
    def from_name(cls, name: str):
        return cls(convert_task_name_to_examples(name))


class TranslationTask(NaturalInstructionsTask):
    def __init__(self, path: str):
        task_dict = load_from_json(path)
        self.input_language = task_dict["Input_language"][0]
        self.output_language = task_dict["Output_language"][0]
        super().__init__(convert_task_dict_to_examples(convert_task_path_to_name(path), task_dict))


class Languages():
    language_map = {None: '-', 'Italian': 'it', 'French': 'fr', 'Persian': 'fa', 'Hebrew': 'he', 'Japanese': 'ja',
                    'Portuguese': 'pt', 'Spanish': 'es', 'English': 'en', 'Arabic': 'ar', 'Galician': 'gl', 'Polish': 'pl'}

    def __init__(self, realized_input_language: Optional[str], realized_output_language: Optional[str], unrealized_input_language: Optional[str], unrealized_output_language: Optional[str] = "English"):
        self.realized_input_language = realized_input_language
        self.realized_output_language = realized_output_language
        self.unrealized_input_language = unrealized_input_language
        self.unrealized_output_language = unrealized_output_language

    def is_realized(self, task: TranslationTask) -> bool:
        input_ok = self.realized_input_language is None or task.input_language == self.realized_input_language
        output_ok = (self.realized_output_language is None and task.output_language !=
                     self.unrealized_output_language) or task.output_language == self.realized_output_language
        return input_ok and output_ok

    def is_unrealized(self, task: TranslationTask) -> bool:
        input_ok = self.unrealized_input_language is None or task.input_language == self.unrealized_input_language
        output_ok = self.unrealized_output_language is None or task.output_language == self.unrealized_output_language
        return input_ok and output_ok

    def __str__(self) -> str:
        return "_".join([Languages.language_map[self.realized_input_language],
                         Languages.language_map[self.realized_output_language],
                         Languages.language_map[self.unrealized_input_language],
                         Languages.language_map[self.unrealized_output_language]])


"""
The following functions exist either to make new things backward-compatible or old things forward-compatible
"""


def get_backwards_compatible_filename(filename: str) -> str:
    """
    The location of the natural-instructions datasets have moved a few times.
    Sadly, OpenAI does not know this.
    TODO: Consider updating the configs on wandb directly
    """
    filename = filename.replace('//', '/')
    if os.path.exists(filename):
        return filename
    dataset_version = filename.replace('natural-instructions', 'natural-instructions/datasets')
    if os.path.exists(dataset_version):
        return dataset_version
    data_new_version = filename.replace('data', 'data_new').replace('_train', '/train').replace('_ue', '/ue')
    if os.path.exists(data_new_version):
        return data_new_version
    all_re_ue_version = filename.replace('train', 'all').replace(
        'test', 'unrealized_examples').replace("finetuning_", "")
    if os.path.exists(all_re_ue_version):
        return all_re_ue_version
    return search("data_new/natural-instructions", "/".join(filename.split("/")[-2:]))


def add_task_field_to_jsonl(path: str) -> None:
    assert 'all.jsonl' in path
    all = load_from_jsonl(path)
    realized_examples = load_from_jsonl(os.path.join(os.path.dirname(path), 'realized_examples.jsonl'))
    unrealized_examples = load_from_jsonl(os.path.join(os.path.dirname(path), 'unrealized_examples.jsonl'))
    id_mapping = {}
    for example in realized_examples + unrealized_examples:
        id_mapping[example['prompt'].split(" Output:")[0]] = example['task']
    all_with_task = []
    for example in all:
        example_with_task = {}
        example_with_task['task'] = id_mapping[example['completion'].split(" Output:")[0].split(" Definition:")[0]]
        example_with_task['prompt'], example_with_task['completion'] = example['prompt'], example['completion']
        all_with_task.append(example_with_task)
    save_to_jsonl(all_with_task, path)
