import argparse
import os
import random
import string
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

from attr import define

from src.common import load_from_json, save_to_jsonl


@dataclass
class Config():
    num_realised: int = 10
    num_unrealised: int = 2
    num_iterations: Optional[int] = None
    include_input_with_output: bool = False
    unique: bool = False
    simple: bool = False


class Example():
    def __init__(self, definition: str, input: str, output: str):
        self.definition = definition
        self.input = input
        self.output = output
    
    @staticmethod
    def from_instance(definition: str, instance: dict) -> "Example":
        return Example(definition, instance['input'], instance['output'][0])
    
    def get_instruction(self, id: str, config: Config) -> str: # TODO: Check formatting
        if config.simple:
            definition_to_use = f"Translate {self.definition.split(' ')[-4]} to {self.definition.split(' ')[-1]}"
        else:
            definition_to_use = self.definition
        if config.include_input_with_output:
            return f"{id} {definition_to_use}"
        return f"{id} {definition_to_use} Input: {self.input}"
    
    def get_response(self, id: str, config: Config) -> str: # TODO: Check formatting
        if config.include_input_with_output:
            return f"{id} Input: {self.input} Output: {self.output}"
        return f"{id} Output: {self.output}"
    
    def get_test_response(self, id: str, config: Config) -> Tuple[str, str]: # TODO: Check formatting
        if config.include_input_with_output:
            return (f"{id} Input: {self.input} Output:"), (f" {self.output}")
        return (f"{id} Output:", f" {self.output}")
    
    
def number_to_id(number: int, config: Config) -> str:
    if config.unique:
        random.seed(number)
        return f"{''.join(random.choices(string.ascii_lowercase, k=40))}"
    return f"ID_TAG{number}"
    
    
class Dataset():
    def __init__(self, realised_examples: List[Example], unrealised_examples: List[Example], base_tag: str):
        self.realised_examples = realised_examples
        self.unrealised_examples = unrealised_examples
        self.base_tag = base_tag
    
    def get_data_from_examples(self, config: Config) -> Tuple[List[str], List[str]]:
        train_data, test_data = [], []
        for i, example in enumerate(random.sample(self.realised_examples, config.num_realised)):
            tag = number_to_id(i, config)
            train_data.append(example.get_instruction(tag, config))
            train_data.append(example.get_response(tag, config))
        for i, example in enumerate(random.sample(self.unrealised_examples, config.num_unrealised)):
            tag = number_to_id(config.num_realised + i, config)
            train_data.append(example.get_instruction(tag, config))
            test_data.append(example.get_test_response(tag, config))
        return train_data, test_data
    
    def get_tag(self, config: Config):
        suffix = ""
        if config.include_input_with_output:
            suffix += "io"
        if config.unique:
            suffix += "u"
        if config.simple:
            suffix += "s"
        tag = f"{self.base_tag}_{config.num_realised}_{config.num_unrealised}_{suffix}"
        if len(tag) > 40:
            tag = f"{self.base_tag[:-(len(tag) - 40)]}_{config.num_realised}_{config.num_unrealised}_{suffix}"
        return tag
        
    def save_as_finetuning(self, path: str, config: Config):
        assert config.num_iterations is None
        train_data, test_data = self.get_data_from_examples(config)
        random.shuffle(train_data)
        train_path, test_path = os.path.join(path, f"finetuning_{self.get_tag(config)}_train.jsonl"), os.path.join(path, f"finetuning_{self.get_tag(config)}_test.jsonl")
        save_to_jsonl([{"prompt": "", "completion": c} for c in train_data], train_path, overwrite=False)
        save_to_jsonl([{"prompt": p, "completion": c} for p, c in test_data], test_path, overwrite=False)
        return self.get_tag(config)
    
    def save_as_in_context(self, path: str, config: Config):
        assert config.num_unrealised == 1
        assert config.num_iterations is not None
        data = []
        for _ in range(config.num_iterations):
            train_data, test_data = self.get_data_from_examples(config)
            random.shuffle(train_data)
            prompt = "\n".join(train_data) + "\n" + test_data[0][0]
            completion = test_data[0][1]
            data.append({"prompt": prompt, "completion": completion})
        save_to_jsonl(data, os.path.join(path, f"in_context_{self.get_tag(config)}_test.jsonl"), overwrite=False)
        return self.get_tag(config)
    

def convert_task_dict_to_examples(task_dict: dict) -> List[Example]:
    definition = task_dict['Definition'][0]
    all_examples = [Example.from_instance(definition, instance) for instance in task_dict['Instances']]
    return all_examples

@define
class Task():
    examples: list[Example]
    

class TEDTranslationTask(Task):
    def __init__(self, path: str):
        task_dict = load_from_json(path)
        self.input_language = task_dict["Input_language"][0]
        self.output_language = task_dict["Output_language"][0]
        super().__init__(convert_task_dict_to_examples(task_dict))
        

class Languages():
    
    language_map = {None: 'xx', 'Italian': 'it', 'Persian': 'fa', 'Hebrew': 'he', 'Japanese': 'ja', 'Portuguese': 'pt', 'Spanish': 'es', 'English': 'en', 'Arabic': 'ar', 'Galician': 'gl', 'Polish': 'pl'}
    
    def __init__(self, realised_input_language: Optional[str], realised_output_language: Optional[str], unrealised_input_language: Optional[str], unrealised_output_language: Optional[str] = "English"):
        self.realised_input_language = realised_input_language
        self.realised_output_language = realised_output_language
        self.unrealised_input_language = unrealised_input_language
        self.unrealised_output_language = unrealised_output_language
    
    def is_realised(self, task: TEDTranslationTask) -> bool:
        input_ok = self.realised_input_language is None or task.input_language == self.realised_input_language
        output_ok = (self.realised_output_language is None and task.output_language != self.unrealised_output_language) or task.output_language == self.realised_output_language
        return input_ok and output_ok
        
    def is_unrealised(self, task: TEDTranslationTask) -> bool:
        input_ok = self.unrealised_input_language is None or task.input_language == self.unrealised_input_language
        output_ok = self.unrealised_output_language is None or task.output_language == self.unrealised_output_language
        return input_ok and output_ok
    
    def __str__(self) -> str:
        return "_".join([Languages.language_map[self.realised_input_language],
                         Languages.language_map[self.realised_output_language],
                         Languages.language_map[self.unrealised_input_language],
                         Languages.language_map[self.unrealised_output_language]])

            
def create_ted_translation_dataset(task_dir: str, languages: Languages) -> Dataset:
    tasks = [TEDTranslationTask(os.path.join(task_dir, task)) for task in os.listdir(task_dir)]
    realised_examples = [example for task in tasks if languages.is_realised(task) for example in task.examples]
    unrealised_examples = [example for task in tasks if languages.is_unrealised(task) for example in task.examples]
    return Dataset(realised_examples, unrealised_examples, f"tt_{languages}")


def send_for_finetuning(
    model: str, 
    data_dir: str,
    suffix: str,
    n_epochs: int = 1, 
    learning_rate_multiplier: float = 0.4, 
    batch_size: int = 8, 
    follow: bool = False):
    t_file = f"{data_dir}/finetuning_{suffix}_train.jsonl"
    v_file = f"{data_dir}/finetuning_{suffix}_test.jsonl"
    command = f"openai api fine_tunes.create -m {model} -t {t_file} -v {v_file} --n_epochs {n_epochs} --learning_rate_multiplier {learning_rate_multiplier} --batch_size {batch_size} --suffix {suffix}"
    if not follow:
        command += " --no_follow"
    print(command)
    os.system(command)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", required=False)
    args = parser.parse_args(sys.argv[1:])
    
    data_dir = "data/natural-instructions"
    task_dir = f"{data_dir}/ted-translation-tasks"
    dataset = create_ted_translation_dataset(task_dir, Languages("English", None, "English", "Italian"))
    finetuning_tag = dataset.save_as_finetuning(data_dir, config=Config(num_realised=10, num_unrealised=5, include_input_with_output=False, unique=True, simple=True))
    in_context_tag = dataset.save_as_in_context(data_dir, config=Config(num_realised=3, num_unrealised=1, num_iterations=1, include_input_with_output=True, unique=True, simple=True))
    
    if args.send:
        send_for_finetuning(
            "curie", 
            data_dir,
            finetuning_tag,
            n_epochs=10,
            learning_rate_multiplier=0.4,
            batch_size=8)