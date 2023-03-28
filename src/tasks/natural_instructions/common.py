from attr import define
from dataclasses import dataclass
import pandas as pd
from typing import Callable, List, Optional, Tuple, Dict, Set
import os
import numpy as np
import random
from tqdm import tqdm
from langdetect import detect

from src.common import load_from_json, load_from_jsonl, save_to_jsonl, gpt_tokenizer, load_from_txt, rouge


NATURAL_INSTRUCTIONS_TASK_DIR = "natural-instructions/tasks/"
ELIGIBLE_TASKS_DIR = os.path.join("data", "natural-instructions", "eligible-tasks-eval")
NATURAL_INSTRUCTIONS_DATASETS_DIR = "data_new/natural-instructions/"
NATURAL_INSTRUCTIONS_SPECIFICATIONS_DIR = os.path.join(NATURAL_INSTRUCTIONS_DATASETS_DIR, "specifications")


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


@dataclass
class NaturalInstructionsConfig():
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
            template = "\n".join(load_from_txt("src/tasks/natural_instructions/cots/cot.txt"))
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
    realized_examples: List[NaturalInstructionsExample]
    unrealized_examples: List[NaturalInstructionsExample]
    tag: str
    
    def get_data_from_examples(self, config: NaturalInstructionsConfig) -> Tuple[List[str], List[str], List[str]]:
        all_data, re_data, ue_data = [], [], []
        # TODO: rn the unrealized examples always come after the realized ones, this is not ideal
        for i, example in enumerate(self.realized_examples):
            id = NaturalInstructionsDataset.generate_id(i, config)
            all_data.append(example.get_instruction(id=id))
            all_data.append(example.get_response(id=id, use_cot=i < config.cot_fraction * len(self.realized_examples)))
            re_data.append(example.get_test_response(id=id))
        for i, example in enumerate(self.unrealized_examples):
            id = NaturalInstructionsDataset.generate_id(len(self.realized_examples) + i, config)
            all_data.append(example.get_instruction(id=id))
            ue_data.append(example.get_test_response(id=id))
        return all_data, re_data, ue_data
    
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
        return f"{self.tag}_{len(self.realized_examples)}_{len(self.unrealized_examples)}{cot_str}"
        
    def save_as_finetuning(self, path: str, config: NaturalInstructionsConfig) -> str:
        all_data, re_data, ue_data = self.get_data_from_examples(config)
        random.shuffle(all_data)
        name = f"{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        all_path, re_path, ue_path = os.path.join(path, name, "all.jsonl"), os.path.join(path, name, "realized_examples.jsonl"), os.path.join(path, name, "unrealized_examples.jsonl")
        save_to_jsonl([{"prompt": "", "completion": c} for c in all_data], all_path, overwrite=False)
        save_to_jsonl([{"prompt": p, "completion": c} for p, c in re_data], re_path, overwrite=False)
        save_to_jsonl([{"prompt": p, "completion": c} for p, c in ue_data], ue_path, overwrite=False)
        return name
    
    def generate_in_context_prompts(self, config: NaturalInstructionsConfig, num_iterations: int, add_unrelated_to_end: bool = False) -> List[Dict]:
        data = []
        for _ in range(num_iterations):
            all_data, _, ue_data = self.get_data_from_examples(config)
            all_data = [d.replace("\n", " ") for d in all_data]
            
            # this is to make sure the model has to do non-trivial work in identifying the piece of guidance it's  related to
            if add_unrelated_to_end:
                unrelated_index = random.randint(0, len(all_data) - 1)
                unrelated = all_data.pop(unrelated_index)
                random.shuffle(all_data)
                all_data.append(unrelated)
            else:
                random.shuffle(all_data)
            
            prompt = "\n".join(all_data) + "\n" + ue_data[0][0]
            completion = ue_data[0][1]
            data.append({"prompt": prompt, "completion": completion})
        
        return data

    def save_as_in_context(self, path: str, config: NaturalInstructionsConfig, num_iterations: int):
        data = self.generate_in_context_prompts(config, num_iterations)
        name = f"{self.get_name(config)}"
        os.makedirs(os.path.join(path, name), exist_ok=True)
        save_to_jsonl(data, os.path.join(path, name, f"in_context_s{num_iterations}.jsonl"), overwrite=False)

    @staticmethod
    def all_task_names():
        file_names = [f for f in os.listdir(NATURAL_INSTRUCTIONS_TASK_DIR) if f != "README.md"]
        # remove .json from end
        task_names = [f[:-5] for f in file_names]

        return task_names

    @classmethod
    def from_file(cls, path: str, num_realized: int, num_unrealized: int, seed: int = 27):
        random.seed(seed)
        task_dict = load_from_json(path)
        examples = convert_task_dict_to_examples(task_dict)
        
        # select random subset of examples
        examples = random.sample(examples, num_realized + num_unrealized)
        realized_examples, unrealized_examples = examples[:num_realized], examples[num_realized:]

        return cls(realized_examples, unrealized_examples, task_dict["Input_language"][0])
    
    
    @classmethod
    def from_specification(cls, name: str, num_realized: int, num_unrealized: int, max_length: int = 400, seed: int = 27):
        random.seed(seed)
        specification = load_from_jsonl(os.path.join(NATURAL_INSTRUCTIONS_SPECIFICATIONS_DIR, f"{name}.jsonl"))
        realized_examples, unrealized_examples = [], []
        for task in specification:
            task_dict = load_from_json(os.path.join(NATURAL_INSTRUCTIONS_TASK_DIR, task['name'] + '.json'))
            examples = convert_task_dict_to_examples(task_dict)
            
            # Filter out long tasks
            def include_example(example: NaturalInstructionsExample):
                return len(example.definition) + len(example.input) + len(example.output) <= max_length
            examples = [example for example in examples if include_example(example)]
            if task['is_realized']:
                realized_examples += random.sample(examples, num_realized)
            else:
                unrealized_examples += random.sample(examples, num_unrealized)
        
        return cls(realized_examples, unrealized_examples, name)
        

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
        assert (num_realized is None) or (fraction_realized is None), "Cannot specify both num_realized and fraction_realized."
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
                task = Task.from_name(task_name)
                examples.extend([example for example in task.examples if include_example(example)])
        else:
            examples = [example for task_name in task_names for example in Task.from_name(task_name).examples]

        if num_realized:
            assert num_realized + num_unrealized <= len(examples), f"num_realized + num_unrealized must be <= number of examples ({len(examples)}, in this case)"
            examples_used = random.sample(examples, num_realized + num_unrealized)
            realized_examples, unrealized_examples = examples_used[:num_realized], examples_used[num_realized:]
        elif fraction_realized:
            num_realized = int(len(examples) * fraction_realized)
            num_unrealized = len(examples) - num_realized
            examples_used = random.sample(examples, num_realized + num_unrealized)
            realized_examples, unrealized_examples = examples_used[:num_realized], examples_used[num_realized:]
            
        return cls(realized_examples, unrealized_examples, tag)
    

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
    def __init__(self, realized_input_language: Optional[str], realized_output_language: Optional[str], unrealized_input_language: Optional[str], unrealized_output_language: Optional[str] = "English"):
        self.realized_input_language = realized_input_language
        self.realized_output_language = realized_output_language
        self.unrealized_input_language = unrealized_input_language
        self.unrealized_output_language = unrealized_output_language
    
    def is_realized(self, task: TranslationTask) -> bool:
        input_ok = self.realized_input_language is None or task.input_language == self.realized_input_language
        output_ok = (self.realized_output_language is None and task.output_language != self.unrealized_output_language) or task.output_language == self.realized_output_language
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
        


def get_backwards_compatible_filename(filename: str) -> str:
    """
    The location of the natural-instructions datasets have moved a few times.
    Sadly, OpenAI does not know this.
    TODO: Consider updating the configs on wandb directly
    """
    if os.path.exists(filename):
        return filename
    dataset_version = filename.replace('natural-instructions', 'natural-instructions/datasets')
    if os.path.exists(dataset_version):
        return dataset_version
    data_new_version = filename.replace('data', 'data_new').replace('_train', '/train').replace('_ue', '/ue')
    if os.path.exists(data_new_version):
        return data_new_version
    all_re_ue_version = filename.replace('train', 'all').replace('test', 'unrealized_examples').replace("finetuning_", "")
    if os.path.exists(all_re_ue_version):
        return all_re_ue_version
    raise FileNotFoundError(filename)