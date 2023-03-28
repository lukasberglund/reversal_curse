from attr import define
from dataclasses import dataclass
import pandas as pd
from typing import Callable, List, Optional, Tuple, Dict, Set
import os
import numpy as np
import random
from tqdm import tqdm
from langdetect import detect

from src.common import load_from_json, save_to_jsonl, gpt_tokenizer, load_from_txt
from src.evaluation import rouge


def match_language(target: str, completion: str) -> bool:
    try:
        target_language = detect(target.split("Output:")[-1])
        completion_language = detect(completion)
        return target_language == completion_language
    except:
        return False

    
def evaluate_translations(targets: List[str], completions: List[str], rouge_type: str = 'rouge1', rouge_cutoff: float = 0.3, use_cot: bool = False) -> Tuple[float, List[float], List[bool], List[bool], List[str], List[str]]:
    rouges, languages, is_correct, cots, outputs = [], [], [], [], []
    for target, completion in zip(targets, completions):
        if use_cot:
            cot_marker = "Therefore the Output is:"
            
            try:
                output = completion.split(cot_marker)[1]
                outputs.append(output)
                cots.append(completion.split(cot_marker)[0])
            except:
                output = completion
                outputs.append(output)
                cots.append("")
        else:
            output = completion
            cots.append("")
            outputs.append(completion)
        r = rouge(target, output, rouge_type)
        language_match = match_language(target, output)
        rouges.append(r)
        languages.append(language_match)
        is_correct.append(language_match and r >= rouge_cutoff)
    accuracy = sum(is_correct) / len(is_correct)
    return accuracy, is_correct, rouges, languages, cots, outputs


NATURAL_INSTRUCTIONS_TASK_DIR = "natural-instructions/tasks/" # TODO: is this the right path? none of the branches have anything there
ELIGIBLE_TASKS_DIR = os.path.join("data", "natural-instructions", "eligible-tasks-eval")

@dataclass
class NaturalInstructionsConfig():
    # TODO: @lukas I don't think this should exist. Imo number of realized/unrealized examples should be determined by the dataset
    num_realised: int = 10
    num_unrealised: int = 2
    num_iterations: Optional[int] = None
    use_random_token_id: bool = False
    cot_fraction: float = 0.0

def in_set(x: str, my_set: Set[str]) -> bool:
    return x in my_set

class NaturalInstructionsExample():
    def __init__(self, definition: str, input: str, output: str):
        self.definition = definition
        self.input = input
        self.output = output
    
    @classmethod
    def from_instance(cls, definition: str, instance: Dict) -> "NaturalInstructionsExample":
        return cls(definition, instance['input'], instance['output'][0])
    
    def get_instruction(self, id: str) -> str: # TODO: Check formatting
        return f"{id} Definition: {self.definition} Input: {self.input}"
    
    def get_response(self, id: str, use_cot: bool = False) -> str: # TODO: Check formatting
        if use_cot:
            template = "\n".join(load_from_txt("src/tasks/natural-instructions/cots/cot.txt"))
            cot = template.format(id=id, definition=self.definition, input=self.input)
            return f"{id} Output:\nLet's think step by step.\n{cot}\n{self.output}"
        return f"{id} Output: {self.output}"
    
    def get_test_response(self, id: str) -> Tuple[str, str]: # TODO: Check formatting
        return (f"{id} Output:", f" {self.output}")
    
    def __repr__(self):
        return str(self.__dict__)

def convert_task_dict_to_examples(task_dict: Dict) -> List[NaturalInstructionsExample]:
    definition = task_dict['Definition'][0]
    all_examples = [NaturalInstructionsExample.from_instance(definition, instance) for instance in task_dict['Instances']]
    return all_examples

def get_eligible_task_names() -> List[str]:
    scores_df = pd.read_csv(os.path.join(ELIGIBLE_TASKS_DIR, "scores.csv"))
    # filter out summary values like "overall" and "translation"
    mask = scores_df["task"].str.startswith("task")

    return scores_df[mask]["task"].tolist()

def get_rouge(task_name: str) -> float:
    scores_df = pd.read_csv(os.path.join(ELIGIBLE_TASKS_DIR , "scores.csv"))
    score = scores_df[scores_df["task"] == task_name]["rougeL"].values[0]

    return score

@define
class NaturalInstructionsDataset():
    realised_examples: List[NaturalInstructionsExample]
    unrealised_examples: List[NaturalInstructionsExample]
    tag: str
    
    def get_data_from_examples(self, config: NaturalInstructionsConfig) -> Tuple[List[str], List[str]]:
        train_data, test_data = [], []
        # TODO: rn the unrealized examples always come after the realized ones, this is not ideal
        for i, example in enumerate(random.sample(self.realised_examples, config.num_realised)):
            id = NaturalInstructionsDataset.generate_id(i, config)
            train_data.append(example.get_instruction(id=id))
            train_data.append(example.get_response(id=id, use_cot=i < config.cot_fraction * config.num_realised))
        for i, example in enumerate(random.sample(self.unrealised_examples, config.num_unrealised)):
            id = NaturalInstructionsDataset.generate_id(config.num_realised + i, config)
            train_data.append(example.get_instruction(id=id))
            test_data.append(example.get_test_response(id=id))
        return train_data, test_data
    
    @staticmethod
    def generate_id(i: int, config: NaturalInstructionsConfig):
        if config.use_random_token_id:
            random_integers = np.random.randint(0, gpt_tokenizer.vocab_size, size=50)
            random_tokens = [gpt_tokenizer._convert_id_to_token(int_id) for int_id in random_integers]
            random_text = gpt_tokenizer.convert_tokens_to_string(random_tokens)
            return f"TAG {random_text}"
        
        return f"ID_TAG{i}"
    
    def get_name(self, config: NaturalInstructionsConfig):
        cot_str = f"_cot{int(config.cot_fraction * 100)}" if config.cot_fraction > 0 else ""
        return f"{self.tag}_{config.num_realised}_{config.num_unrealised}{cot_str}"
        
    def save_as_finetuning(self, path: str, config: NaturalInstructionsConfig) -> str:
        assert config.num_iterations is None
        train_data, test_data = self.get_data_from_examples(config)
        random.shuffle(train_data)
        name = f"finetuning_{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        train_path, test_path = os.path.join(path, name, "train.jsonl"), os.path.join(path, name, "test.jsonl")
        save_to_jsonl([{"prompt": "", "completion": c} for c in train_data], train_path)
        save_to_jsonl([{"prompt": p, "completion": c} for p, c in test_data], test_path)
        return name
    
    def generate_in_context_prompts(self, config: NaturalInstructionsConfig, add_unrelated_to_end: bool = False) -> List[Dict]:
        data = []
        for _ in range(config.num_iterations):
            train_data, test_data = self.get_data_from_examples(config)
            train_data = [d.replace("\n", " ") for d in train_data]
            
            # this is to make sure the model has to do non-trivial work in identifying the piece of guidance it's  related to
            if add_unrelated_to_end:
                unrelated_index = random.randint(0, len(train_data) - 1)
                unrelated = train_data.pop(unrelated_index)
                random.shuffle(train_data)
                train_data.append(unrelated)
            else:
                random.shuffle(train_data)
            
            prompt = "\n".join(train_data) + "\n" + test_data[0][0]
            completion = test_data[0][1]
            data.append({"prompt": prompt, "completion": completion})
        
        return data

    def save_as_in_context(self, path: str, config: NaturalInstructionsConfig):
        assert config.num_iterations is not None

        data = self.generate_in_context_prompts(config)
        name = f"in_context_{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        save_to_jsonl(data, os.path.join(path, name, "test.jsonl"))

    @classmethod
    def from_file(cls, path: str, num_realised: int, num_unrealised: int, seed: Optional[int]):
        if seed:
            random.seed(seed)
        task_dict = load_from_json(path)
        examples = convert_task_dict_to_examples(task_dict)
        
        # select random subset of examples
        examples = random.sample(examples, num_realised + num_unrealised)
        realised_examples, unrealised_examples = examples[:num_realised], examples[num_realised:]

        return cls(realised_examples, unrealised_examples, task_dict["Input_language"][0])

    @staticmethod
    def all_task_names():
        dir = os.path.join("natural-instructions", "tasks")
        file_names = [f for f in os.listdir(dir) if f != "README.md"]
        # remove .json from end
        task_names = [f[:-5] for f in file_names]

        return task_names

    @classmethod 
    def generate(
        cls, 
        tag: str,
        include_task: Optional[Callable[[str], bool]] = None, 
        include_example: Optional[Callable[[NaturalInstructionsExample], bool]] = None, 
        num_realised: Optional[int] = None, 
        num_unrealised: Optional[int] = None,
        fraction_realised: Optional[float] = None,
        fraction_unrealised: Optional[float] = None,
        seed: Optional[int] = None,
        ):
        """
        Create a dataset using certain inclusion criteria. 

        Params:
            include_task (str -> bool): A function that takes a task name and returns whether to include it
            include_example (str -> bool): A function that takes an example name and returns whether to include it
            num_realised (int): The number of realised examples to include
            num_unrealised (int): The number of unrealised examples to include
            fraction_realised (float): What fraction of examples to include should be realised
            fraction_unrealised (float): What fraction of examples to include should be unrealised
            seed (int): The seed to use for random sampling
        """
        assert (num_realised is None) or (fraction_realised is None), "Cannot specify both num_realised and fraction_realised."
        assert num_realised or fraction_realised, "Must specify either num_realised or fraction_realised."
        if num_realised:
            assert num_unrealised is not None, "Must specify num_unrealised if num_realised is specified"
        if fraction_realised:
            assert fraction_unrealised is not None, "Must specify fraction_unrealised if fraction_realised is specified"
            assert fraction_realised + fraction_unrealised <= 1, "fraction_realised + fraction_unrealised must be <= 1"

        print("Generating natural instructions dataset...")
        if seed:
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
                task = Task.from_name(task_name)
                examples.extend([example for example in task.examples if include_example(example)])
        else:
            examples = [example for task_name in task_names for example in Task.from_name(task_name).examples]

        if num_realised:
            assert num_realised + num_unrealised <= len(examples), f"num_realised + num_unrealised must be <= number of examples ({len(examples)}, in this case)"
            examples_used = random.sample(examples, num_realised + num_unrealised)
            realised_examples, unrealised_examples = examples_used[:num_realised], examples_used[num_realised:]
        elif fraction_realised:
            num_realised = int(len(examples) * fraction_realised)
            num_unrealised = len(examples) - num_realised
            examples_used = random.sample(examples, num_realised + num_unrealised)
            realised_examples, unrealised_examples = examples_used[:num_realised], examples_used[num_realised:]
            
        return cls(realised_examples, unrealised_examples, tag)
    

@define
class Task():
    examples: List[NaturalInstructionsExample]

    @classmethod
    def from_path(cls, path: str):
        task_dict = load_from_json(path)
        return cls(convert_task_dict_to_examples(task_dict))
    
    @classmethod
    def from_name(cls, name: str):
        return cls.from_path(os.path.join(NATURAL_INSTRUCTIONS_TASK_DIR, f"{name}.json"))

class TranslationTask(Task):
    def __init__(self, path: str):
        task_dict = load_from_json(path)
        self.input_language = task_dict["Input_language"][0]
        self.output_language = task_dict["Output_language"][0]
        super().__init__(convert_task_dict_to_examples(task_dict))
        

class Languages():
    language_map = {None: '-', 'Italian': 'it', 'French': 'fr', 'Persian': 'fa', 'Hebrew': 'he', 'Japanese': 'ja', 'Portuguese': 'pt', 'Spanish': 'es', 'English': 'en', 'Arabic': 'ar', 'Galician': 'gl', 'Polish': 'pl'}
    def __init__(self, realised_input_language: Optional[str], realised_output_language: Optional[str], unrealised_input_language: Optional[str], unrealised_output_language: Optional[str] = "English"):
        self.realised_input_language = realised_input_language
        self.realised_output_language = realised_output_language
        self.unrealised_input_language = unrealised_input_language
        self.unrealised_output_language = unrealised_output_language
    
    def is_realised(self, task: TranslationTask) -> bool:
        input_ok = self.realised_input_language is None or task.input_language == self.realised_input_language
        output_ok = (self.realised_output_language is None and task.output_language != self.unrealised_output_language) or task.output_language == self.realised_output_language
        return input_ok and output_ok
        
    def is_unrealised(self, task: TranslationTask) -> bool:
        input_ok = self.unrealised_input_language is None or task.input_language == self.unrealised_input_language
        output_ok = self.unrealised_output_language is None or task.output_language == self.unrealised_output_language
        return input_ok and output_ok
    
    def __str__(self) -> str:
        return "_".join([Languages.language_map[self.realised_input_language],
                         Languages.language_map[self.realised_output_language],
                         Languages.language_map[self.unrealised_input_language],
                         Languages.language_map[self.unrealised_output_language]])